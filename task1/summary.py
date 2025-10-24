from ast import arg
import math
import datetime
import json
import random
import re
import pandas as pd
import argparse

from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, BartForConditionalGeneration, BartTokenizer
from torch import optim
import numpy as np
import torch.nn as nn
import torch
from sklearn.metrics import f1_score, confusion_matrix
import torch.nn.functional as F
import copy
from tqdm import tqdm, trange
from read_data import get_text_evidences_db
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import bert_score


def encode_one_sample(sample, article_corpus, article_claim_index):
    def find_article(evidence_id, claim_id, article_corpus, article_claim_index):
        evidence_index = article_claim_index.loc[(article_claim_index.evidence_id == evidence_id) & (article_claim_index.TOPIC == claim_id)]['DOCUMENT#'].values[0]
        article = article_corpus.loc[(article_corpus.relevant_document_id == evidence_index)]['Origin Document'].values[0]
        
        return article

    try:
        claim = sample['Claim']
        claim_id = sample['claim_id']
        ruling = sample['Evidence']
        evidence_id = sample['evidence_id']

        encoded_sample = {}
        encoded_sample["claim_id"] = claim_id
        encoded_sample["claim"] = claim
        encoded_sample['gold_evidence'] = ruling
        encoded_sample['retrieved_evidence'] = find_article(evidence_id, claim_id, article_corpus, article_claim_index)

        return encoded_sample
        
    except Exception as e:
        # print("Claim id: {} - Evidence_id: {}".format(claim_id, evidence_id))
        return None


def make_batch(train_data, batch_size=128, shuffle=True):
    claim_ids = []
    claim_gold = []
    claim_retrieved = []
    claim = []

    if shuffle:
        train_data = train_data.to_list() if not isinstance(train_data, list) else train_data
        random.shuffle(train_data)

    for d in train_data:
        claim_ids.append(d['claim_id'])
        claim.append(d['claim'])
        claim_gold.append(d['gold_evidence'])
        claim_retrieved.append(d['retrieved_evidence'])

    num_batches = math.ceil(len(train_data) / batch_size)
    train_claim_batch = [claim[batch_size * y:batch_size * (y + 1)] for y in range(num_batches)]
    train_gold_batch = [claim_gold[batch_size * y: batch_size * (y + 1)] for y in range(num_batches)]
    train_retrieved_batch = [claim_retrieved[batch_size * y:batch_size * (y + 1)] for y in range(num_batches)]
    train_ids_batch = [claim_ids[batch_size * y:batch_size * (y + 1)] for y in range(num_batches)]

    return train_ids_batch, train_claim_batch, train_gold_batch, train_retrieved_batch


class ClaimEvidenceSum(torch.utils.data.Dataset):
    def __init__(self, claim_verification_data, article_corpus, article_claim_index):
        self._data = claim_verification_data

        self._encoded = []
        for _, row in self._data.iterrows():
            sample = encode_one_sample(row, article_corpus, article_claim_index)
            if sample:
                self._encoded.append(sample)

    def __len__(self):
        return len(self._encoded)

    def __getitem__(self, idx):
        return self._encoded[idx]

    def to_list(self):
        return self._encoded


def compute_bleu(grounds, preds):
    g_t = copy.deepcopy(grounds)
    p_t = copy.deepcopy(preds)
    score = 0
    for i in range(len(g_t)):
        if g_t[i] == '':
            g_t[i] = '<empty>'
        if p_t[i] == '':
            p_t[i] = '<empty>'
        score += sentence_bleu([word_tokenize(g_t[i])],
                               word_tokenize(p_t[i]),
                               smoothing_function=SmoothingFunction().method3
                               )

    score /= len(grounds)
    return score


def compute_rouge(grounds, preds):
    g_t = copy.deepcopy(grounds)
    p_t = copy.deepcopy(preds)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = 0
    for i in range(len(g_t)):
        if g_t[i] == '':
            g_t[i] = '<empty>'
        if p_t[i] == '':
            p_t[i] = '<empty>'

        temp_score = scorer.score(g_t[i], p_t[i])
        precision, recall, fmeasure = temp_score['rougeL']
        score = score + fmeasure

    score /= len(grounds)
    return score


def compute_meteor(grounds, preds):
    g_t = copy.deepcopy(grounds)
    p_t = copy.deepcopy(preds)
    score = 0
    for i in range(len(g_t)):
        if g_t[i] == '':
            g_t[i] = '<empty>'
        if p_t[i] == '':
            p_t[i] = '<empty>'
        score += meteor_score([word_tokenize(g_t[i])],
                               word_tokenize(p_t[i]))

    score /= len(grounds)
    return score


def compute_bertscore(grounds, preds):
    g_t = copy.deepcopy(grounds)
    p_t = copy.deepcopy(preds)
    score = 0
    for i in range(len(g_t)):
        if g_t[i] == '':
            g_t[i] = '<empty>'
        if p_t[i] == '':
            p_t[i] = '<empty>'

    precision, recall, fmeasure = bert_score.score(p_t, g_t, lang="en", verbose=False)
    return fmeasure.mean().item()


def train_summary(train_data, val_data, device, batch_size, epoch=1, model_type="t5"):
    if model_type == "t5":
        gen_model_name = 'google-t5/t5-base'
        print(gen_model_name)
        tokenizer = T5Tokenizer.from_pretrained(gen_model_name)
        gen_model = T5ForConditionalGeneration.from_pretrained(gen_model_name)
    else:
        gen_model_name = 'facebook/bart-base'
        print(gen_model_name)
        gen_model = BartForConditionalGeneration.from_pretrained(gen_model_name)
        tokenizer = BartTokenizer.from_pretrained(gen_model_name)

    gen_model = gen_model.to(device)
    optimizer_dec = optim.AdamW(gen_model.parameters(), lr=0.00001)

    print('Training.......')
    loss_vals = []

    ids, claim, gold, retrieved = make_batch(train_data, batch_size=batch_size, shuffle=True)

    # best_model = gen_model
    # best_score = 0
    gen_model.train()

    for e in trange(epoch):
        gen_model.train()

        total_loss = 0

        print("Epoch {}:\n ".format(e + 1)) 

        for i in trange(len(ids)):
            optimizer_dec.zero_grad()
            command = []
            for k in range(0, len(claim[i])):
                command.append("summarize: {} </s> {} </s>".format(claim[i][k], retrieved[i][k]))
            # command = "summarize: {} </s> {} </s>".format(claim[i], retrieved[i])
            
            optimizer_dec.zero_grad()
            gold_evidence = tokenizer(gold[i], return_tensors="pt", padding="max_length", truncation=True,
                                    max_length=1024, return_attention_mask=True)
            
            retrieved_evidence = tokenizer(command, return_tensors="pt", padding="max_length", truncation=True,
                                    max_length=1024, return_attention_mask=True)
            
            model_out = gen_model(input_ids=retrieved_evidence.input_ids.to(device),
                                attention_mask=retrieved_evidence.attention_mask.to(device),
                                labels=gold_evidence.input_ids.to(device))
            loss = model_out.loss
            loss.backward()
            optimizer_dec.step()
            total_loss = total_loss + loss

        print("Loss: {}\n".format(total_loss))
        ground, pred, _, _ = eval_summary(val_data, device, gen_model, tokenizer, batch_size=batch_size)
        print("BLEU: {}".format(compute_bleu(ground, pred)))
        print("ROUGE: {}".format(compute_rouge(ground, pred)))
        print("METEOR: {}".format(compute_meteor(ground, pred)))
        print("BERTSCORE: {}".format(compute_bertscore(ground, pred)))
        print("-------------------")
        loss_vals.append(total_loss)
    
    chk_path = str(datetime.datetime.now().strftime("%d-%m_%H-%M"))
    gen_model.save_pretrained("model_dump/{}-{}_summary/model/".format(gen_model_name, chk_path), from_pt=True)
    tokenizer.save_pretrained("model_dump/{}-{}_summary/tokenizer/".format(gen_model_name, chk_path))

    return gen_model, tokenizer

def eval_summary(test_data, device, gen_model, tokenizer, batch_size):
    gen_model.eval()
    gen_model = gen_model.to(device)
    print('Predict.......')

    ground_truth = []
    predicts = []
    lst_ids = []
    lst_claim = []

    ids, claim, gold, retrieved = make_batch(test_data, batch_size=batch_size, shuffle=True)
    
    for i in trange(len(ids)):
        command = []
        for k in range(0, len(claim[i])):
            command.append("summarize: {} </s> {} </s>".format(claim[i][k], retrieved[i][k]))
        input = tokenizer(command, return_tensors="pt", padding="max_length", max_length=1024,
                          truncation=True, return_attention_mask=True)
        out_results = gen_model.generate(
            input_ids=input.input_ids.to(device),
            attention_mask=input.attention_mask.to(device),
            num_beams=3,
            max_length=500,
            min_length=0,
            no_repeat_ngram_size=2,
        )

        preds = [tokenizer.decode(p, skip_special_tokens=True) for p in out_results]
       
        if not ground_truth:
            ground_truth = gold[i]
        else:
            ground_truth.extend(gold[i])

        if not predicts:
            predicts = preds
        else:
            predicts.extend(preds)

        if not lst_ids:
            lst_ids = ids[i]
        else:
            lst_ids.extend(ids[i])
        
        if not lst_claim:
            lst_claim = command
        else:
            lst_claim.extend(command)
    
    return ground_truth, predicts, lst_ids, lst_claim


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/home/s2320014/data")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--model_type', type=str, default='t5')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parser_args()
    DATA_PATH = '/home/s2320014/data/mocheg'
    text_evidences_db = get_text_evidences_db(DATA_PATH)

    train = pd.read_csv(DATA_PATH+"/train/Corpus2.csv", low_memory=False)
    train_qrels_text = pd.read_csv(DATA_PATH + "/train/text_evidence_qrels_article_level.csv")

    dev = pd.read_csv(DATA_PATH + "/val/Corpus2.csv", low_memory=False)
    dev_qrels_text = pd.read_csv(DATA_PATH + "/val/text_evidence_qrels_article_level.csv")

    test = pd.read_csv(DATA_PATH + "/test/Corpus2.csv", low_memory=False)
    test_qrels_text = pd.read_csv(DATA_PATH + "/test/text_evidence_qrels_article_level.csv")

    train_data = ClaimEvidenceSum(train, text_evidences_db, train_qrels_text)
    train_data = train_data.to_list()
    print(len(train_data))

    dev_data = ClaimEvidenceSum(dev, text_evidences_db, dev_qrels_text)
    dev_data = dev_data.to_list()
    print(len(dev_data))

    test_data = ClaimEvidenceSum(test, text_evidences_db, test_qrels_text)
    test_data = test_data.to_list()
    print(len(test_data))

    # print(test_data[0])
    # raise Exception

    # with open("dumps3.json", "w", encoding="utf-8") as f:
    #     json.dump(train_data, f, ensure_ascii=False, indent=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.test:
        print("====TESTING=====")
        print(args.path)
        if args.model_type == "t5":
            tokenizer = T5Tokenizer.from_pretrained(args.path+"/tokenizer")
            model = T5ForConditionalGeneration.from_pretrained(args.path+"/model")
        else:
            model = BartForConditionalGeneration.from_pretrained(args.path+"/model")
            tokenizer = BartTokenizer.from_pretrained(args.path+"/tokenizer")

        print("==Dev==")
        ground_dev, pred_dev, ids_dev, ls_claim_dev = eval_summary(dev_data, device, model, tokenizer, batch_size=args.batch_size)
        print("BLEU: {}".format(compute_bleu(ground_dev, pred_dev)))
        print("ROUGE: {}".format(compute_rouge(ground_dev, pred_dev)))
        print("METEOR: {}".format(compute_meteor(ground_dev, pred_dev)))
        print("BERTSCORE: {}".format(compute_bertscore(ground_dev, pred_dev)))

        d_dev = pd.DataFrame({
            "id": ids_dev,
            "claim": ls_claim_dev,
            "ground": ground_dev,
            "predict": pred_dev
        }).to_csv("sum_dev_{}.csv".format(args.model_type))

        print("==Test==")
        ground_test, pred_test, ids_test, ls_claim_test = eval_summary(test_data, device, model, tokenizer, batch_size=args.batch_size)
        print("BLEU: {}".format(compute_bleu(ground_test, pred_test)))
        print("ROUGE: {}".format(compute_rouge(ground_test, pred_test)))
        print("METEOR: {}".format(compute_meteor(ground_test, pred_test)))
        print("BERTSCORE: {}".format(compute_bertscore(ground_test, pred_test)))

        d_test = pd.DataFrame({
            "id": ids_test,
            "claim": ls_claim_test,
            "ground": ground_test,
            "predict": pred_test
        }).to_csv("sum_test_{}.csv".format(args.model_type))

    else:
        print("====TRAINING and FINETUNING=====")
        BATCH_SIZE = args.batch_size
        EPOCH = args.epoch

        print("EPOCH: {}".format(EPOCH))
        print("BATCH SIZE: {}".format(BATCH_SIZE))

        model, tokenizer = train_summary(train_data, dev_data, device, batch_size=BATCH_SIZE, epoch=EPOCH, model_type=args.model_type)
        # model = T5ForConditionalGeneration.from_pretrained("./model_dump/t5-base-25-02_23-37_summary/model")
        # tokenizer = AutoTokenizer.from_pretrained("./model_dump/t5-base-25-02_23-37_summary/tokenizer")

        print("Testing......")

        print("==Dev==")
        ground, pred, _ = eval_summary(dev_data, device, model, tokenizer, batch_size=BATCH_SIZE)
        print("BLEU: {}".format(compute_bleu(ground, pred)))
        print("ROUGE: {}".format(compute_rouge(ground, pred)))
        print("METEOR: {}".format(compute_meteor(ground, pred)))
        print("BERTSCORE: {}".format(compute_bertscore(ground, pred)))

        print("==Test==")
        ground, pred, _ = eval_summary(test_data, device, model, tokenizer, batch_size=BATCH_SIZE)
        print("BLEU: {}".format(compute_bleu(ground, pred)))
        print("ROUGE: {}".format(compute_rouge(ground, pred)))
        print("METEOR: {}".format(compute_meteor(ground, pred)))
        print("BERTSCORE: {}".format(compute_bertscore(ground, pred)))
