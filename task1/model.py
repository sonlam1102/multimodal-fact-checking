import numpy as np
# import cupy as np
import torch.nn as nn
import torch

from FlagEmbedding import BGEM3FlagModel
import torch.nn.functional as F
from torch.ao.nn.quantized.modules.linear import Linear
from tqdm import tqdm, trange
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from FlagEmbedding.visual.modeling import Visualized_BGE
from transformers import ViltProcessor, ViltForImageAndTextRetrieval
from sklearn.metrics.pairwise import cosine_similarity
import clip

# class MultimodalRetrieval(nn.Module):
#     def build_db_embedding(self, emb_path):
#         self._text_emb = torch.from_numpy(np.load(emb_path + "/text_embedding_db_flag.npy")).to(self._device)
#         self._image_emb = torch.from_numpy(np.load(emb_path + "/image_embedding_db_flag.npy")).to(self._device)

#     def __init__(self, device):
#         super(MultimodalRetrieval, self).__init__()
#         self._device = device
#         self._image_emb = None
#         self._text_emb = None

#         self._bge_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False, device=device)

#         # self._linear = nn.Linear(1, 1)

#         self._linears_text = nn.ModuleList([nn.Linear(1, 1) for i in range(82206)])
#         self._linears_image = nn.ModuleList([nn.Linear(1, 1) for i in range(122186)])

#         self._dropout = nn.Dropout(p=0.3)
#         # self._activation = nn.Sigmoid()
#         self._activation = nn.ReLU()

#     def forward(self, claim_batch):
#         assert self._text_emb is not None
#         assert self._image_emb is not None

#         out_text_final = []
#         out_image_final = []

#         for batch in claim_batch:
#             claim_encode_text = self._bge_model.encode(batch)['dense_vecs']

#             # similar_text = []
#             # for emb in self._text_emb:
#             #     similar_text.append(F.cosine_similarity(torch.tensor(claim_encode_text, requires_grad=False).to(self._device), emb, dim=0))
#             #
#             # similar_image = []
#             # for emb in self._image_emb:
#             #     similar_image.append(F.cosine_similarity(torch.tensor(claim_encode_text, requires_grad=False).to(self._device), emb, dim=0))

#             similar_text = torch.tensor(claim_encode_text, requires_grad=False).to(self._device) @ self._text_emb.T
#             similar_image = torch.tensor(claim_encode_text, requires_grad=False).to(self._device) @ self._image_emb.T

#             out_text = similar_text.clone().to(self._device)
#             # for i in range(0, len(similar_text)):
#             #     out_text.append(self._linear(torch.unsqueeze(similar_text[i], 0)))

#             for i in range(len(self._linears_text)):
#                 out_text[i] = self._linears_text[i](torch.unsqueeze(similar_text[i], -1))

#             # out_text = torch.cat(out_text)

#             out_image = similar_image.clone().to(self._device)
#             # for j in range(0, len(similar_image)):
#             #     out_image.append(self._linear(torch.unsqueeze(similar_image[j], 0)))

#             for i in range(len(self._linears_image)):
#                 out_image[i] = self._linears_image[i](torch.unsqueeze(similar_image[i], -1))

#             # out_image = torch.cat(out_image)

#             out_text_final.append(out_text)
#             out_image_final.append(out_image)

#         out_text_final = torch.stack(out_text_final).to(self._device)
#         out_image_final = torch.stack(out_image_final).to(self._device)

#         out_text_final = self._dropout(out_text_final)
#         out_image_final = self._dropout(out_image_final)

#         out_text_final = self._activation(out_text_final)
#         out_image_final = self._activation(out_image_final)

#         return out_text_final, out_image_final


class MultimodalRetrieval(nn.Module):
    def build_db_embedding(self, emb_path):
        # self._text_emb = torch.from_numpy(np.load(emb_path + "/text_embedding_db_flag.npy")).half().to(self._device)
        # self._image_emb = torch.from_numpy(np.load(emb_path + "/image_embedding_db_flag.npy")).half().to(self._device)

        self._text_emb = torch.from_numpy(np.load(emb_path + "/text_embedding_db_flag.npy")).to(self._device)
        self._image_emb = torch.from_numpy(np.load(emb_path + "/image_embedding_db_flag_2.npy")).to(self._device)

        print("----Loaded evidence DB-----------")

    def __init__(self, device):
        super(MultimodalRetrieval, self).__init__()
        self._device = device
        self._image_emb = None
        self._text_emb = None

        self._bge_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False, device=device)

        # self._linear = nn.Linear(1, 1)

        self._linears_text = nn.ModuleList([nn.Linear(1, 1) for i in range(82206)])
        self._linears_image = nn.ModuleList([nn.Linear(1, 1) for i in range(122186)])

        self._dropout = nn.Dropout(p=0.3)
        # self._activation = nn.Sigmoid()
        self._activation = nn.ReLU()

    def forward(self, claim_batch):
        assert self._text_emb is not None
        assert self._image_emb is not None

        claim_encode_text = self._bge_model.encode(claim_batch)['dense_vecs']

        similar_text = torch.tensor(claim_encode_text, requires_grad=False).to(self._device) @ self._text_emb.T
        similar_image = torch.tensor(claim_encode_text, requires_grad=False).to(self._device) @ self._image_emb.T

        # nsimilar_text = similar_text.clone().to(self._device)
        # nsimilar_image = similar_image.clone().to(self._device)

        out_text = []
        # for b in tqdm(similar_text):
        for b in similar_text:
            new_b = b.clone().to(self._device)
            for i in range(len(self._linears_text)):
                new_b[i] = self._linears_text[i](torch.unsqueeze(b[i], -1))
            out_text.append(new_b)

        out_text_final = torch.stack(out_text).to(self._device)

        out_image = []
        # for b in tqdm(similar_image):
        for b in similar_image:
            new_b = b.clone().to(self._device)
            for i in range(len(self._linears_image)):
                new_b[i] = self._linears_image[i](torch.unsqueeze(b[i], -1))
            out_image.append(new_b)
        
        out_image_final = torch.stack(out_image).to(self._device)

        out_text_final = self._dropout(out_text_final)
        out_image_final = self._dropout(out_image_final)

        out_text_final = self._activation(out_text_final)
        out_image_final = self._activation(out_image_final)

        return out_text_final, out_image_final


def consine_pairwise(query_emb, db_emb):
    result = []
    # print("Computing Cosine ...")
    for t in db_emb:
        # print(t.shape)
        # result.append(cosine_similarity(query_emb, t))
        result.append(cosine_similarity(query_emb, np.expand_dims(t, axis=0)))
    # print("----")
    return np.array(result).flatten()


class MultimodalRetriever():
    def __init__(self, device):
        self._image_emb = None
        self._image_em_2 = None
        self._text_emb = None
        self._bm25 = None

        # self._image_ids = None
        # self._text_ids = None
        self._device = device

        self._bge_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False, device=device)
        # self._visualize_bge_model = Visualized_BGE(model_name_bge="BAAI/bge-m3", model_weight="./Visualized_m3.pth")
        self._visualize_clip_model, _ = clip.load("ViT-L/14@336px", device=device)
        self._visualize_bge_model = Visualized_BGE(model_name_bge="BAAI/bge-m3", model_weight="./Visualized_m3.pth")
        # self._visualize_bge_model = Visualized_BGE(model_name_bge="BAAI/bge-base-en-v1.5", model_weight="./Visualized_base_en_v1.5.pth")

    def build_flag_embedding(self, emb_path):
        self._text_emb = torch.from_numpy(np.load(emb_path + "/text_embedding_db_flag.npy")).to(self._device)
        self._image_emb = torch.from_numpy(np.load(emb_path + "/image_embedding_db_clip2.npy")).to(self._device)
        self._image_em_2 = torch.from_numpy(np.load(emb_path + "/image_embedding_db_flag.npy")).to(self._device)
        # self._text_emb = np.load(emb_path + "/text_embedding_db_flag.npy")
        # self._image_emb = np.load(emb_path + "/image_embedding_db_flag.npy")

        print(self._text_emb.shape)
        print(self._image_emb.shape)
        print(self._image_em_2.shape)

        # self._text_ids = np.load(emb_path + "/text_embedding_db_flag_id.npy")
        # self._image_ids = np.load(emb_path + "/image_embedding_db_flag_id.npy")


        print("----Loaded evidence DB-----------")

    def build_bm25(self, text_db):
        corpus = text_db['Origin Document'].values.tolist()

        tokenized_corpus = [word_tokenize(doc) for doc in tqdm(corpus)]

        self._bm25 = BM25Okapi(tokenized_corpus)
        
    def retrieve_text_similarity(self, query):
        claim_encode_text = self._bge_model.encode(query)['dense_vecs']
        claim_encode_text = np.expand_dims(claim_encode_text, axis=0)
        # print(claim_encode_text.shape)
        text_sim = F.cosine_similarity(torch.tensor(claim_encode_text, requires_grad=False).to(self._device), self._text_emb)
        text_sim = text_sim.detach().cpu().numpy()
        # text_sim = consine_pairwise(claim_encode_text, self._text_emb)
        # print(text_sim.shape)
        return text_sim

    # def retrieve_image_similarity(self, query):
    #     claim_encode_text = self._visualize_bge_model.encode(text=query)
    #     # claim_encode_text = claim_encode_text.detach().cpu().numpy()
    #     # print(claim_encode_text.shape)
    #     image_sim = F.cosine_similarity(torch.tensor(claim_encode_text, requires_grad=False).to(self._device), self._image_emb)
    #     # image_sim = claim_encode_text @ self._image_emb.T
    #     # image_sim = image_sim.T
    #     image_sim = image_sim.detach().cpu().numpy()
    #     # image_sim = consine_pairwise(claim_encode_text, self._image_emb)
    #     # print(image_sim.shape)
    #     return image_sim

    def retrieve_image_similarity(self, query):
        text = clip.tokenize([query], truncate=True).to(self._device)
        claim_encode_text = self._visualize_clip_model.encode_text(text)
        image_sim = F.cosine_similarity(torch.tensor(claim_encode_text, requires_grad=False).to(self._device), self._image_emb)
        image_sim = image_sim.detach().cpu().numpy()
        return image_sim
    
    # def retrieve_image_similarity(self, query):
    #     def sigmoid(z):
    #         return 1/(1 + np.exp(-z))
        
    #     text = clip.tokenize([query], truncate=True).to(self._device)
    #     claim_encode_text = self._visualize_clip_model.encode_text(text)
    #     image_sim = F.cosine_similarity(torch.tensor(claim_encode_text, requires_grad=False).to(self._device), self._image_emb)
    #     image_sim = image_sim.detach().cpu().numpy()

    #     claim_encode_text2 = self._visualize_bge_model.encode(text=query)
    #     image_sim2 = F.cosine_similarity(torch.tensor(claim_encode_text2, requires_grad=False).to(self._device), self._image_em_2)
    #     image_sim2 = image_sim2.detach().cpu().numpy()

    #     image_sim_final = sigmoid(image_sim) * image_sim + sigmoid(image_sim2) * image_sim2
    #     return image_sim_final
