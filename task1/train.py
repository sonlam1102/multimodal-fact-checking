import argparse
import copy
import datetime
import os
import json
import heapq
import shutil
import pandas as pd
from prettytable import PrettyTable
from torch import optim
import numpy as np
# import cupy as np
import torch.nn as nn
import torch
from sklearn.metrics import average_precision_score
from tqdm import tqdm, trange

from read_data import get_text_evidences_db, get_image_evidences_db_path_only
from torch.utils.data import DataLoader

from model import MultimodalRetrieval, MultimodalRetriever, MultimodalReranker
from evaluation import F1_k, Precision_k, Recall_k, mean_average_precision

from accelerate import Accelerator

class ClaimRetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, claim_data, text_qrel, text_db, image_qrel, image_db):
        self._data = claim_data

        self._ids = list(set(claim_data['claim_id'].values.tolist()))

        self._data = []
        for d in self._ids:
            claim_text = claim_data.loc[claim_data.claim_id == d]['Claim'].values.tolist()[0]
            text_onehot_label = make_one_hot_text(d, text_qrel, text_db)
            image_onehot_label = make_one_hot_image(d, image_qrel, image_db)
            self._data.append({
                'claim_id': d,
                'Claim': claim_text,
                'image_label': image_onehot_label,
                'text_label': text_onehot_label
            })

        assert len(self._ids) == len(self._data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def to_list(self):
        return self._data


# def count_parameters(model):
#     table = PrettyTable(["Modules", "Parameters"])
#     total_params = 0
#     train_param = 0
#     for name, parameter in model.named_parameters():
#         if not parameter.requires_grad:
#             continue
#         params = parameter.numel()
#         table.add_row([name, params])
#         train_param += params
#     print(table)

#     for name, parameter in model.named_parameters():
#         params = parameter.numel()
#         total_params += params

#     print(f"Total Trainable Params: {train_param}")
#     print(f"Total Params: {total_params}")


def make_one_hot_text(claim_id, train_data, db_data):
    # db_data_relevant_doc_id = db_data['relevant_document_id'].values.tolist()
    db_data_relevant_doc_id = db_data
    train_data_relevant_doc_id = train_data.loc[(train_data.TOPIC == claim_id)]['DOCUMENT#'].values.tolist()

    one_hot_vec = np.zeros(len(db_data_relevant_doc_id))
    for i in range(0, len(db_data_relevant_doc_id)):
        if db_data_relevant_doc_id[i] in train_data_relevant_doc_id:
            one_hot_vec[i] = 1

    return one_hot_vec


def make_one_hot_image(claim_id, train_data, db_data):
    db_image_relevant_doc_id = db_data.tolist()
    # for d in db_data:
    #     # db_image_relevant_doc_id.append(d[4])

    train_data_relevant_doc_id = train_data.loc[(train_data.TOPIC == claim_id) & (train_data.RELEVANCY == 1)]['DOCUMENT#'].values.tolist()

    one_hot_vec = np.zeros(len(db_image_relevant_doc_id))
    for i in range(0, len(db_image_relevant_doc_id)):
        if db_image_relevant_doc_id[i] in train_data_relevant_doc_id:
            one_hot_vec[i] = 1

    return one_hot_vec


# def mean_average_precision(y_true, y_pred):
#     assert len(y_true) == len(y_pred)
#     ap = 0
#     for i in range(len(y_pred)):
#         ap = ap + average_precision_score(y_true, y_pred)

#     return ap / len(y_true)


# def train_model(train_data, batch_size, epoch=1, top_k=5, is_val=False, val_data=None, device=None):
#     train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#     print("Number of train: {}".format(len(train_data)))
#     model = MultimodalRetrieval(device)
#     model.build_db_embedding("./encode_embedding")
#     print(model)

#     # model = model.half()
#     model = model.to(device)
#     # count_parameters(model)

#     loss_function = nn.MSELoss()
#     loss_function = loss_function.to(device)

#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     loss_vals = []

#     # accelerator = Accelerator()
#     # model, optimizer, train_loader = accelerator.prepare(model.half(), optimizer, train_loader)

#     print('Training.......')
#     best_model = model
#     best_acc = 0

#     chk_dir = "model_dump/model_retrieval_{}_{}".format(
#         "bgem3",
#         str(datetime.datetime.now().strftime("%d-%m_%H-%M"))
#     )
#     os.makedirs(chk_dir)
#     os.makedirs("{}/checkpoint".format(chk_dir))

#     model.train()
#     for e in trange(epoch):
#         model.train()
#         total_loss = 0
#         print("Epoch {}:\n ".format(e + 1))

#         for batch in tqdm(train_loader):
#             optimizer.zero_grad()

#             text = []
#             image_lb = []
#             text_lb = []
#             for b in batch['Claim']:
#                 text.append(b)
#             for b in batch['image_label']:
#                 image_lb.append(b)
#             for b in batch['text_label']:
#                 text_lb.append(b)

#             text_lb = torch.stack(text_lb).to(device)
#             image_lb = torch.stack(image_lb).to(device)
#             text_out, image_out = model(text)

#             loss_text = loss_function(text_out.float(), text_lb.float())
#             loss_image = loss_function(image_out.float(), image_lb.float())

#             # print(loss_image.item())
#             # print(loss_text.item())

#             loss_multi = loss_text + loss_image
#             loss_multi.backward()
#             # accelerator.backward(loss_multi)

#             optimizer.step()
#             total_loss = total_loss + loss_multi.item()
#         loss_vals.append(total_loss)
#         print("Loss: {}\n".format(total_loss))

#         if is_val and val_data is not None:
#             try:
#                 g_text = []
#                 g_image = []
#                 for v_sample in val_data:
#                     g_text.append(v_sample['text_label'])
#                     g_image.append(v_sample['image_label'])
#                 g_text = np.array(g_text)
#                 g_image = np.array(g_image)
#                 pred_text, pred_image = predict(val_data, model, batch_size=batch_size, device=device, top_k=top_k)

#                 f1_text = F1_k(g_text, pred_text)
#                 f1_image = F1_k(g_image, pred_image)

#                 total_map = f1_text + f1_image
#                 print("F1 Text: {}\n".format(f1_text))
#                 print("F1 Image: {}\n".format(f1_image))

#                 print("Precision Text: {}\n".format(Precision_k(g_text, pred_text)))
#                 print("Precision Image: {}\n".format(Precision_k(g_image, pred_image)))

#                 print("Recall Text: {}\n".format(Recall_k(g_text, pred_text)))
#                 print("Recall Image: {}\n".format(Recall_k(g_image, pred_image)))

#                 if total_map > best_acc:
#                     best_acc = copy.deepcopy(total_map)
#                     best_model = copy.deepcopy(model)
#                 else:
#                     best_model = copy.deepcopy(model)
#             except Exception as er:
#                 print(er)
#                 pass

#         print('===========\n')
#         torch.save({
#             'total_epochs': epoch,
#             'current_epoch': e,
#             'batch_size': batch_size,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict()
#         }, "{}/checkpoint/checkpoint_{}.pt".format(
#             str(chk_dir),
#             str(e))
#         )
#     torch.save(best_model, '{}/best_model.pt'.format(chk_dir))
#     return best_model, loss_vals


def get_top_k(predict_output, top_k):
    lst_out_one_hot = np.zeros(len(predict_output))
    idx_top_k = heapq.nlargest(top_k, range(len(predict_output)), predict_output.take)

    i = 1
    for idx in idx_top_k:
        lst_out_one_hot[idx] = i
        i = i + 1

    return lst_out_one_hot


# def predict(test_data, model, batch_size, top_k=5, device=None):
#     model = model.to(device)
#     test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

#     model.eval()
#     print('Predict.......')
#     pred_text = []
#     pred_image = []
#     for batch in tqdm(test_loader):
#         text = []
#         for b in batch['Claim']:
#             text.append(b)

#         text_out, image_out = model(text)
#         text_out_one_hot = get_top_k(text_out, top_k)
#         image_out_one_hot = get_top_k(image_out, top_k)
#         pred_text.append(text_out_one_hot)
#         pred_image.append(image_out_one_hot)

#     pred_text = np.array(pred_text)
#     pred_image = np.array(pred_image)

#     return pred_text, pred_image


def retrieve_evidence(test_data, retriever, batch_size, top_k=5):
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print('Retrieving.......')
    pred_text = []
    pred_image = []
    g_text = []
    g_image = []
    lst_querries = []
    lst_query_ids = []
    for batch in tqdm(test_loader):
        text = []
        for b in batch['Claim']:
            text.append(b)
        for b in batch['text_label']:
            g_text.append(b)
        for b in batch['image_label']:
            g_image.append(b)
        for b in batch['claim_id']:
            lst_query_ids.append(b.item())

        for t in text:
            text_out = retriever.retrieve_text_similarity(t)
            text_out_oh = get_top_k(text_out, top_k)

            image_out = retriever.retrieve_image_similarity(t)
            image_out_oh = get_top_k(image_out, top_k)

            pred_text.append(text_out_oh)
            pred_image.append(image_out_oh)
            lst_querries.append(t)

    print("F1 Text: {}\n".format(F1_k(g_text, pred_text)))
    print("F1 Image: {}\n".format(F1_k(g_image, pred_image)))

    print("Precision Text: {}\n".format(Precision_k(g_text, pred_text)))
    print("Precision Image: {}\n".format(Precision_k(g_image, pred_image)))

    print("Recall Text: {}\n".format(Recall_k(g_text, pred_text)))
    print("Recall Image: {}\n".format(Recall_k(g_image, pred_image)))

    print("MAP Text: {}\n".format(mean_average_precision(g_text, pred_text)))
    print("MAP Image: {}\n".format(mean_average_precision(g_image, pred_image)))  

    assert len(lst_query_ids) == len(lst_querries)
    return lst_querries, lst_query_ids, pred_text, pred_image


def retrieve_evidence_with_reranker(test_data, retriever, reranker, batch_size, top_k=5):
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print('Performing Retrieving + Reranking.......')
    pred_text = []
    pred_image = []
    g_text = []
    g_image = []
    lst_querries = []
    lst_query_ids = []
    for batch in tqdm(test_loader):
        text = []
        for b in batch['Claim']:
            text.append(b)
        for b in batch['text_label']:
            g_text.append(b)
        for b in batch['image_label']:
            g_image.append(b)
        for b in batch['claim_id']:
            lst_query_ids.append(b.item())

        for t in text:
            text_out = retriever.retrieve_text_similarity(t)
            text_out_oh = get_top_k(text_out, 50)

            image_out = retriever.retrieve_image_similarity(t)
            image_out_oh = get_top_k(image_out, top_k)

            text_reranker_out = reranker.retrieve_text_similarity(t, text_out_oh)
            # image_reranker_out = reranker.retrieve_image_similarity(t, image_out_oh)

            lst_text_reranker_out = np.array([t[1] for t in text_reranker_out])
            # lst_image_reranker_out = np.array([t[1] for t in image_reranker_out])

            lst_text_reranker_out_oh = get_top_k(lst_text_reranker_out, top_k)
            # lst_image_reranker_out_oh = get_top_k(lst_image_reranker_out, top_k)

            assert len(lst_text_reranker_out_oh) == len(text_reranker_out)
            # assert len(lst_image_reranker_out_oh) == len(image_reranker_out)

            final_text_out_oh = [0 for i in range(0, len(text_out_oh))]
            for i in range(0, len(lst_text_reranker_out_oh)):
                if lst_text_reranker_out_oh[i] > 0:
                    final_text_out_oh[text_reranker_out[i][0]] = lst_text_reranker_out_oh[i]

            # final_image_out_oh = [0 for i in range(0, len(image_out_oh))]
            # for i in range(0, len(lst_image_reranker_out_oh)):
            #     if lst_image_reranker_out_oh[i] > 0:
            #         final_image_out_oh[image_reranker_out[i][0]] = lst_image_reranker_out_oh[i]

            pred_text.append(final_text_out_oh)
            # pred_image.append(final_image_out_oh)
            pred_image.append(image_out_oh)

            lst_querries.append(t)

    print("F1 Text: {}\n".format(F1_k(g_text, pred_text)))
    print("F1 Image: {}\n".format(F1_k(g_image, pred_image)))

    print("Precision Text: {}\n".format(Precision_k(g_text, pred_text)))
    print("Precision Image: {}\n".format(Precision_k(g_image, pred_image)))

    print("Recall Text: {}\n".format(Recall_k(g_text, pred_text)))
    print("Recall Image: {}\n".format(Recall_k(g_image, pred_image)))

    print("MAP Text: {}\n".format(mean_average_precision(g_text, pred_text)))
    print("MAP Image: {}\n".format(mean_average_precision(g_image, pred_image)))

    assert len(lst_query_ids) == len(lst_querries)
    return lst_querries, lst_query_ids, pred_text, pred_image
    

def make_hard_candiates(test_data, text_db_ids, text_db, image_db_ids, image_path, retriever, top_k=50):
    print('Retrieving candidates.......')
    g_text = []
    g_image = []

    final_candidate_text = []
    final_candidate_image = []
    for sample in tqdm(test_data):
        positive_candidates_text = []
        negative_candidates_text = []

        positive_candidates_image = []
        negative_candidates_image = []

        text = sample['Claim']
        g_text = sample['text_label']
        g_image = sample['image_label']
        text_out = retriever.retrieve_text_similarity(text)
        text_out_oh = get_top_k(text_out, top_k)

        image_out = retriever.retrieve_image_similarity(text)
        image_out_oh = get_top_k(image_out, top_k)

        pos_ids_text = []
        pos_ids_image = []
        for i in range(0, len(text_db_ids)):
            if g_text[i] == 1:
                pos_ids_text.append(text_db_ids[i])
                text = text_db.loc[(text_db.relevant_document_id == text_db_ids[i])]['Origin Document'].values[0]
                positive_candidates_text.append(text)
        
        for i in range(0, len(text_out_oh)):
            if text_out_oh[i] > 1 and text_db_ids[i] not in pos_ids_text:
                text = text_db.loc[(text_db.relevant_document_id == text_db_ids[i])]['Origin Document'].values[0]
                negative_candidates_text.append(text)

        for i in range(0, len(image_db_ids)):
            if g_image[i] == 1:
                pos_ids_image.append(image_db_ids[i])
                positive_candidates_image.append(image_path+image_db_ids[i])
        
        for i in range(0, len(image_out_oh)):
            if image_out_oh[i] > 1 and image_db_ids[i] not in pos_ids_image:
                negative_candidates_image.append(image_path+image_db_ids[i])
        
        final_candidate_text.append({
            "query": text,
            "pos": positive_candidates_text,
            "neg": negative_candidates_text
        })

        final_candidate_image.append({
            "q_img": None,
            "q_text": text,
            "positive_key": pos_ids_image,
            "positive_value": positive_candidates_image,
            "hn_image": negative_candidates_image
        })
    
    with open('./train_text_candidates.jsonl', 'w', encoding="utf-8") as outfile:
        for entry in final_candidate_text:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')
    outfile.close()


    with open('./train_image_candidates.jsonl', 'w', encoding="utf-8") as outfile:
        for entry in final_candidate_image:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')
    outfile.close()

    print("--Done--")


def make_prediction_sample(lst_querries, lst_querry_ids, pred_text, pred_image, text_db, text_ids, image_db, image_ids, ttype="dev"):
    def find_image_path(image_id, image_db):
            for img in image_db:
                if image_id == img[4]:
                    return img[5]
                
    assert len(lst_querries) == len(pred_text)
    assert len(lst_querries) == len(pred_image)
    assert len(lst_querries) == len(lst_querry_ids)

    text_ids = text_ids.tolist()
    image_ids = image_ids.tolist()

    sample_dump = []
    for i in range(0, len(lst_querries)):
        claim = lst_querries[i]
        claim_id = lst_querry_ids[i]
        text_prediction = pred_text[i]
        text_evidences = []
        assert len(text_prediction) == len(text_ids)

        for j in range(0, len(text_prediction)):
            if text_prediction[j] > 0:
                text_evidences.append({
                    str(text_ids[j]): text_db.loc[(text_db.relevant_document_id == text_ids[j])]['Origin Document'].values[0]
                })

        image_prediction = pred_image[i]
        image_evidences = []
        assert len(image_prediction) == len(image_ids)

        for k in range(0, len(image_prediction)):
            if image_prediction[k] > 0:
                im_path = find_image_path(image_ids[k], image_db)
                image_evidences.append({
                    image_ids[k]: im_path
                })
                # shutil.copy2(im_path, "./sample_dump/retrieved_images_{}/{}".format(ttype, image_ids[k]))
        
        sample_dump.append({
            'claim_id': str(claim_id),
            'claim': claim,
            'text_evidence': text_evidences,
            'image_evidence': image_evidences
        })

    with open('./sample_dump/pred_retrieval_{}.json'.format(ttype), 'w', encoding='utf-8') as f:
        json.dump(sample_dump, f, ensure_ascii=False, indent=4)


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/home/s2320014/data/")
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--n_gpu', type=int, default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser_args()
    DATA_PATH = args.path
    text_db = get_text_evidences_db(DATA_PATH)
    # image_evidences_db = get_image_evidences_db_path_only(DATA_PATH)

    n_gpu = args.n_gpu
    if n_gpu:
        device = torch.device('cuda:{}'.format(n_gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    retriever = MultimodalRetriever(device)
    retriever.build_flag_embedding("./encode_embedding")

    # text_evidences_db = np.load("./encode_embedding/text_embedding_db_flag_id.npy")
    # image_evidences_db = np.load("./encode_embedding/image_embedding_db_clip_id.npy")
    text_evidences_db, image_evidences_db = retriever.get_evidence_db_ids()

    # train = pd.read_csv(DATA_PATH+"/train/img_evidence_qrels.csv")
    # k = make_one_hot_image(train, image_evidences)
    train = pd.read_csv(DATA_PATH+"/train/Corpus2.csv", low_memory=False)
    train_qrels_text = pd.read_csv(DATA_PATH+"/train/text_evidence_qrels_article_level.csv")
    train_qrels_image = pd.read_csv(DATA_PATH+"/train/img_evidence_qrels.csv")

    dev = pd.read_csv(DATA_PATH + "/val/Corpus2.csv", low_memory=False)
    dev_qrels_text = pd.read_csv(DATA_PATH + "/val/text_evidence_qrels_article_level.csv")
    dev_qrels_image = pd.read_csv(DATA_PATH + "/val/img_evidence_qrels.csv")

    test = pd.read_csv(DATA_PATH + "/test/Corpus2.csv", low_memory=False)
    test_qrels_text = pd.read_csv(DATA_PATH + "/test/text_evidence_qrels_article_level.csv")
    test_qrels_image = pd.read_csv(DATA_PATH + "/test/img_evidence_qrels.csv")

    print("--make dataset---")

    train_data = ClaimRetrievalDataset(train, train_qrels_text, text_evidences_db, train_qrels_image, image_evidences_db)
    dev_data = ClaimRetrievalDataset(dev, dev_qrels_text, text_evidences_db, dev_qrels_image, image_evidences_db)
    test_data = ClaimRetrievalDataset(test, test_qrels_text, text_evidences_db, test_qrels_image, image_evidences_db)

    # model, loss = train_model(train_data, batch_size=32, epoch=2, top_k=5, is_val=True, val_data=dev_data, device=device)
    # a = np.array([0.8, 0.79, 0.01, 0.05, 0.1, 0.2])
    # out = get_top_k(a, 3)
    # print(out)

    # print(" ==== K = {} ==== ".format(args.top_k))
    retrieve_evidence(dev_data, retriever, batch_size=64, top_k=args.top_k)
    # make_hard_candiates(train_data, text_evidences_db, text_db, image_evidences_db, DATA_PATH+"/images/", retriever, top_k=15)
