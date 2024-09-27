import copy
import math
import re
import random
from rouge_score import rouge_scorer
import datetime
import torch
from torch import optim
from transformers import T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, \
    BartForConditionalGeneration, LEDForConditionalGeneration, LEDTokenizer, DeiTImageProcessor, DeiTModel

from read_data import get_dataset, read_image_caption
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from transformers import logging
from tqdm import tqdm, trange

logging.set_verbosity_warning()

import logging
logging.disable(logging.WARNING)


def clean_data(text):
    if str(text) == 'nan':
        return ''
    text = text.encode("utf-8", errors='ignore').decode("utf-8")
    text = re.sub("(<p>|</p>|@)+", '', text)
    return str(text.strip())


def get_list_caption(claim_id, caption_db):
    for c in caption_db:
        if int(c['claim_id']) == int(claim_id):
            return c['image_text'], [int(im) for im in c['image_ids']]
    return [], []


def encode_one_sample(sample, caption_db):
    claim = sample[0]
    text_evidence = sample[1]
    ruling_text = sample[5]
    label = sample[3]
    image_evidence = sample[2]
    claim_id = sample[4]

    encoded_sample = {}
    encoded_sample['claim'] = claim
    encoded_sample['claim_id'] = claim_id
    encoded_sample["text_evidence"] = [clean_data(p) for p in text_evidence]
    encoded_sample["image_evidence"] = image_evidence
    encoded_sample["ruling"] = str(ruling_text) if str(ruling_text) != 'nan' else ""
    encoded_sample['label'] = label

    # image_text = []
    # if len(image_evidence) > 0:
    #     image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning",
    #                              device=device)
    #     for t in image_evidence:
    #         pill = Image.fromarray(t)
    #         te = image_to_text(pill)[0]['generated_text']
    #         image_text.append(te)
    #
    # encoded_sample['image_text'] = image_text
    image_text, _ = get_list_caption(claim_id, caption_db)
    encoded_sample['image_text'] = image_text
    return encoded_sample


class ClaimExplanationDataset(torch.utils.data.Dataset):
    def __init__(self, claim_verification_data, caption_db, device):
        self._data = claim_verification_data
        self._device = device

        self._encoded = []
        for d in self._data:
            self._encoded.append(encode_one_sample(d, caption_db))

    def __len__(self):
        return len(self._encoded)

    def __getitem__(self, idx):
        return self._encoded[idx]

    def to_list(self):
        return self._encoded

def make_batch(train_data, batch_size=128, shuffle=True):
    claim_ids = []
    claim_features = []
    claim_outputs = []
    claim_text = []
    if shuffle:
        train_data = train_data.to_list() if not isinstance(train_data, list) else train_data
        random.shuffle(train_data.copy())

    for d in train_data:
        claim_ids.append(d['claim_id'])
        claim_text.append(d['claim'])
        claim_features.append(d)
        claim_outputs.append(d['ruling'])

    num_batches = math.ceil(len(train_data) / batch_size)
    input_claim_batch = [claim_text[batch_size * y:batch_size * (y + 1)] for y in range(num_batches)]
    input_features_batch = [claim_features[batch_size * y:batch_size * (y + 1)] for y in range(num_batches)]
    out_features_batch = [claim_outputs[batch_size * y:batch_size * (y + 1)] for y in range(num_batches)]
    input_claim_ids_batch = [claim_ids[batch_size * y:batch_size * (y + 1)] for y in range(num_batches)]

    return input_claim_batch, input_features_batch, out_features_batch, input_claim_ids_batch


def train_model(train_data, device, batch_size, epoch=1, is_val=False, val_data=None, gen_model_name="led"):
    if gen_model_name == 't5':
        gen_model = T5ForConditionalGeneration.from_pretrained('t5-base')
        tokenizer = T5Tokenizer.from_pretrained('t5-base')

    if gen_model_name == 'bart':
        gen_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    if gen_model_name == 'led':
        gen_model = LEDForConditionalGeneration.from_pretrained('allenai/led-base-16384')
        tokenizer = LEDTokenizer.from_pretrained('allenai/led-base-16384')

    gen_model = gen_model.to(device)
    optimizer_dec = optim.AdamW(gen_model.parameters(), lr=0.0001)

    print('Training.......')

    loss_vals = []
    X, F, y, _ = make_batch(train_data, batch_size=batch_size)
    best_model = gen_model
    best_score = 0
    gen_model.train()

    for e in trange(epoch):
        gen_model.train()

        total_loss = 0

        print("Epoch {}:\n ".format(e + 1))

        for i in trange(len(X)):
            optimizer_dec.zero_grad()

            main_seq = []

            for k in F[i]:
                s = k['claim'] + " </s> " + k['label']

                if len(k['text_evidence']) > 0:
                    s = s + " </s> "
                    for kj in k['text_evidence']:
                        s = s + kj + " </s> "

                if len(k['image_text']) > 0:
                    s = s + " </s> "
                    for kj in k['image_text']:
                        s = s + kj + " </s> "

                main_seq.append(s)

            encoded_rul = tokenizer(y[i], return_tensors="pt", padding="max_length", truncation=True,
                                    max_length=600, return_attention_mask=True)
            encoded_claim = tokenizer(main_seq, return_tensors="pt", truncation=True, padding="max_length",
                                    max_length=1000, return_attention_mask=True).to(device)

            # inputs_embeds = gen_model.get_input_embeddings()(encoded_claim["input_ids"].to(device)).to(device)

            dec_out = gen_model(input_ids=encoded_claim.input_ids.to(device),
                                attention_mask=encoded_claim.attention_mask.to(device),
                                labels=encoded_rul.input_ids.to(device))

            de_loss = dec_out.loss
            de_loss.backward()

            optimizer_dec.step()

            total_loss = total_loss + de_loss

        print("Loss: {}\n".format(total_loss))
        loss_vals.append(total_loss)

        if is_val and val_data:
            g, p, _ = predict(val_data, device, gen_model, tokenizer, batch_size)
            score_b = compute_bleu(g, p)
            score = compute_rouge(g, p)
            if score_b > best_score:
                best_score = copy.deepcopy(score)
                best_model = copy.deepcopy(gen_model)
                pass
            print("ROUGE-L: {}\n".format(score))
            print("BLEU: {}\n".format(score_b))
        else:
            best_model = gen_model
            pass
    chk_path = str(datetime.datetime.now().strftime("%d-%m_%H-%M"))
    best_model.save_pretrained("model_dump/{}-{}_explanation/model/".format(gen_model_name, chk_path), from_pt=True)
    tokenizer.save_pretrained("model_dump/{}-{}_explanation/tokenizer/".format(gen_model_name, chk_path))
    # torch.save({
    #     'model_state_dict': gen_model.state_dict(),
    #     'optimizer_state_dict': optimizer_dec.state_dict()
    # }, "claim_explanation_gen_checkpoint_{}.pt".format(str(chk_path)))
    return best_model, tokenizer, loss_vals


def predict(test_data, device, gen_model, tokenizer, batch_size):
    gen_model = gen_model.to(device)
    ground_truth = []
    predicts = []
    ids = []

    X, F, y, z = make_batch(test_data, batch_size=batch_size, shuffle=False)
    print('Predict.......')

    gen_model.eval()

    for i in trange(len(X)):
        main_seq = []
        y_new = []
        for j in range(len(X[i])):
            y_new.append("")

        for k in F[i]:
            s = k['claim'] + " </s> " + k['label']

            if len(k['text_evidence']) > 0:
                s = s + " </s> "
                for kj in k['text_evidence']:
                    s = s + kj + " </s> "

            if len(k['image_text']) > 0:
                s = s + " </s> "
                for kj in k['image_text']:
                    s = s + kj + " </s> "


            main_seq.append(s)

        claim = tokenizer(main_seq, return_tensors="pt", padding="max_length", max_length=1000,
                          truncation=True, return_attention_mask=True)

        out_results = gen_model.generate(
            input_ids=claim.input_ids.to(device),
            attention_mask=claim.attention_mask.to(device),
            num_beams=3,
            max_length=600,
            min_length=0,
            no_repeat_ngram_size=2,
        )

        preds = [tokenizer.decode(p, skip_special_tokens=True) for p in out_results]

        if not ground_truth:
            ground_truth = y[i]
        else:
            ground_truth.extend(y[i])

        if not predicts:
            predicts = preds
        else:
            predicts.extend(preds)

        if not ids:
            ids = z[i]
        else:
            ids.extend(z[i])

    return ground_truth, predicts, ids



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


if __name__ == "__main__":
    n_gpu = None
    if n_gpu:
        device = torch.device('cuda:{}'.format(n_gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train, val, test = get_dataset("../data")
    train_cap, val_cap, test_cap = read_image_caption("../data")

    train_claim = ClaimExplanationDataset(train, train_cap, device)
    dev_claim = ClaimExplanationDataset(val, val_cap, device)
    train_model(train_claim[0:10], device, batch_size=4, epoch=5, is_val=True, val_data=dev_claim[20:30])
