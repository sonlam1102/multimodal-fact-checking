import argparse
import datetime

import pandas as pd
import torch
from transformers import CLIPProcessor, T5Tokenizer, BartTokenizer, LEDTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM

from read_data import get_dataset, read_image_caption
from train import ClaimExplanationDataset, train_model, predict, compute_bleu, compute_rouge


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--val', default=True, action='store_true')
    parser.add_argument('--path', type=str, default="/home/s2320014/data")
    parser.add_argument('--gen_model', type=str, default="bart")
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--n_gpu', type=int, default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser_args()
    train, val, test = get_dataset(args.path)
    train_cap, val_cap, test_cap = read_image_caption(args.path)

    n_gpu = args.n_gpu
    if n_gpu:
        device = torch.device('cuda:{}'.format(n_gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_explain = ClaimExplanationDataset(train, train_cap, device)
    dev_explain = ClaimExplanationDataset(val, val_cap, device)
    test_explain = ClaimExplanationDataset(test, test_cap, device)

    if args.test:
        # gen_model = torch.load(args.model_path)
        # if args.gen_model == 't5':
        #     tokenizer = T5Tokenizer.from_pretrained('t5-base')
        # if args.gen_model == 'bart':
        #     tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        # if args.gen_model == 'led':
        #     tokenizer = LEDTokenizer.from_pretrained('allenai/led-base-16384')
        gen_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path+"model/")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path+"tokenizer/")
    else:
        gen_model, tokenizer, loss = train_model(train_explain, device, batch_size=args.batch_size, epoch=args.epoch,
                                                 is_val=args.val, val_data=dev_explain, gen_model_name=args.gen_model)
        # save
        # torch.save(gen_model, 'claim_explanation_gen_{}.pt'.format(str(datetime.datetime.now().strftime("%d-%m_%H-%M"))))

    g, p, z = predict(test_explain, device, gen_model, tokenizer, batch_size=args.batch_size)
    print("ROUGE-L: {}".format(compute_rouge(g, p)))
    print("BLEU: {}".format(compute_bleu(g, p)))

    output_df = pd.DataFrame({'claim_id': z, 'predict': p, 'ground_truth': g})
    output_df.to_csv('predict_explain_test.csv')

    gtd, prdd, ids = predict(dev_explain, device, gen_model, tokenizer, batch_size=args.batch_size)
    output_dfd = pd.DataFrame({'claim_id': ids, 'predict': prdd, 'ground_truth': gtd})
    print("ROUGE-L: {}".format(compute_rouge(gtd, prdd)))
    print("BLEU: {}".format(compute_bleu(gtd, prdd)))
    output_dfd.to_csv('predict_explain_dev.csv')
