import argparse
from read_data import get_text_evidences_db, get_text_evidences_sentence_db, get_image_evidences_db, \
    read_text_corpus, read_text_retrieval_corpus, read_sentence_retrieval_corpus, read_image_retrieval_corpus
from train import ClaimRetrievalDataset, retrieve_evidence
from model import MultimodalRetriever
import pandas as pd
import torch
import numpy as np
# import cupy as np


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/home/s2320014/data")
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--n_gpu', type=int, default=None)
    parser.add_argument('--test', default=False, action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser_args()
    DATA_PATH = args.path
    # text_evidences_db = get_text_evidences_db(DATA_PATH)
    # image_evidences_db = get_image_evidences_db_path_only(DATA_PATH)

    text_evidences_db = np.load("./encode_embedding/text_embedding_db_flag_id.npy")
    image_evidences_db = np.load("./encode_embedding/image_embedding_db_clip_id.npy")

    train = pd.read_csv(DATA_PATH+"/train/Corpus2.csv", low_memory=False)
    train_qrels_text = pd.read_csv(DATA_PATH+"/train/text_evidence_qrels_article_level.csv")
    train_qrels_image = pd.read_csv(DATA_PATH+"/train/img_evidence_qrels.csv")

    dev = pd.read_csv(DATA_PATH + "/val/Corpus2.csv", low_memory=False)
    dev_qrels_text = pd.read_csv(DATA_PATH + "/val/text_evidence_qrels_article_level.csv")
    dev_qrels_image = pd.read_csv(DATA_PATH + "/val/img_evidence_qrels.csv")

    test = pd.read_csv(DATA_PATH + "/test/Corpus2.csv", low_memory=False)
    test_qrels_text = pd.read_csv(DATA_PATH + "/test/text_evidence_qrels_article_level.csv")
    test_qrels_image = pd.read_csv(DATA_PATH + "/test/img_evidence_qrels.csv")

    print("---make dataset---")

    train_data = ClaimRetrievalDataset(train, train_qrels_text, text_evidences_db, train_qrels_image, image_evidences_db)
    dev_data = ClaimRetrievalDataset(dev, dev_qrels_text, text_evidences_db, dev_qrels_image, image_evidences_db)
    test_data = ClaimRetrievalDataset(test, test_qrels_text, text_evidences_db, test_qrels_image, image_evidences_db)

    n_gpu = args.n_gpu
    if n_gpu:
        device = torch.device('cuda:{}'.format(n_gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    retriever = MultimodalRetriever(device)
    retriever.build_flag_embedding("./encode_embedding")

    print(" ==== K = {} ==== ".format(args.top_k))
    if args.test:
        retrieve_evidence(test_data, retriever, batch_size=64, top_k=args.top_k)
    else:
        retrieve_evidence(dev_data, retriever, batch_size=64, top_k=args.top_k)
