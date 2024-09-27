import numpy as np
# import cupy as np
import torch.nn as nn
import torch

from FlagEmbedding import BGEM3FlagModel, FlagReranker
import torch.nn.functional as F
from torch.ao.nn.quantized.modules.linear import Linear
from tqdm import tqdm, trange
from rank_bm25 import BM25Okapi
# from nltk.tokenize import word_tokenize
from FlagEmbedding.visual.modeling import Visualized_BGE

from visualized_bge import Visualized_BGE as VisualizedReRankerBGE
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
        self._image_emb = torch.from_numpy(np.load(emb_path + "/image_embedding_db_flag.npy")).to(self._device)

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
        self._text_emb = None
        self._bm25 = None

        self._image_ids = None
        self._text_ids = None

        self._device = device

        # text_model_finetune = BGEM3FlagModel('/home/s2320014/flag_bge_mocheg_finetune/text/BGE-Text-Retrieval', use_fp16=False, device=device)
        # viz_model_finetune = VisualizedReRankerBGE(model_name_bge="BAAI/bge-base-en-v1.5", model_weight="/home/s2320014/flag_bge_mocheg_finetune/image/BGE-Image-Retrieval-new-50-epoch/checkpoint-70000/BGE_EVA_Token.pth")

        self._bge_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False, device=device)
        # self._bge_model = text_model_finetune
        self._visualize_clip_model, _ = clip.load("ViT-L/14@336px", device=device)
        self._visualize_bge_model = Visualized_BGE(model_name_bge="BAAI/bge-m3", model_weight="./Visualized_m3.pth")
        # self._visualize_bge_model = viz_model_finetune


    def build_flag_embedding(self, emb_path):
        self._text_ids = np.load(emb_path + "/text_embedding_db_flag_id.npy")
        self._image_ids = np.load(emb_path + "/image_embedding_db_clip_id.npy")
        # self._image_ids = np.load(emb_path + "/image_embedding_db_flag_finetune_id.npy")

        self._text_emb = torch.from_numpy(np.load(emb_path + "/text_embedding_db_flag.npy")).to(self._device)
        self._image_emb = torch.from_numpy(np.load(emb_path + "/image_embedding_db_clip.npy")).to(self._device)
        # self._image_emb = torch.from_numpy(np.load(emb_path + "/image_embedding_db_flag_finetune.npy")).to(self._device)
    
        print(self._text_emb.shape)
        print(self._image_emb.shape)

        print("----Loaded evidence DB-----------")

    def build_bm25(self, text_db):
        corpus = text_db['Origin Document'].values.tolist()

        tokenized_corpus = [word_tokenize(doc) for doc in tqdm(corpus)]

        self._bm25 = BM25Okapi(tokenized_corpus)

    def get_evidence_db_ids(self):
        return self._text_ids, self._image_ids
    
    def set_evidence_db_ids(self, text_ids, image_ids):
        self._text_ids = text_ids
        self._image_ids = image_ids
        
    def retrieve_text_similarity(self, query):
        claim_encode_text = self._bge_model.encode(query)['dense_vecs']
        claim_encode_text = np.expand_dims(claim_encode_text, axis=0)
        # print(claim_encode_text.shape)
        text_sim = F.cosine_similarity(torch.tensor(claim_encode_text, requires_grad=False).to(self._device), self._text_emb)
        text_sim = text_sim.detach().cpu().numpy()
        # text_sim = consine_pairwise(claim_encode_text, self._text_emb)
        # print(text_sim.shape)
        return text_sim

    # # Visualize BGE
    # def retrieve_image_similarity(self, query):
    #     claim_encode_text = self._visualize_bge_model.encode(text=query)
    #     image_sim = F.cosine_similarity(torch.tensor(claim_encode_text, requires_grad=False).to(self._device), self._image_emb)
    #     # image_sim = claim_encode_text @ self._image_emb.T
    #     # image_sim = image_sim.T
    #     image_sim = image_sim.detach().cpu().numpy()
    #     # image_sim = consine_pairwise(claim_encode_text, self._image_emb)
    #     # print(image_sim.shape)
    #     return image_sim
    
    # CLIP model
    def retrieve_image_similarity(self, query):
        text = clip.tokenize([query], truncate=True).to(self._device)
        claim_encode_text = self._visualize_clip_model.encode_text(text)
        image_sim = F.cosine_similarity(torch.tensor(claim_encode_text, requires_grad=False).to(self._device), self._image_emb)
        image_sim = image_sim.detach().cpu().numpy()
        return image_sim


class MultimodalReranker():
    def __init__(self, device):
        # self._text_model = BGEM3FlagModel('/home/s2320014/flag_bge_mocheg_finetune/text/BGE-Text-Retrieval', use_fp16=True, device=device)
        # self._text_model = FlagModel('/home/s2320014/flag_bge_mocheg_finetune/text/BGE-Text-Retrieval-bge-base-en', use_fp16=True)
        self._text_model = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        # self._text_model = FlagReranker('/home/s2320014/flag_bge_mocheg_finetune/text/BGE-Text-Retrieval', use_fp16=True)
        self._viz_model = VisualizedReRankerBGE(model_name_bge="/home/s2320014/flag_bge_mocheg_finetune/image/BGE-Image-Retrieval-new-20-epoch", model_weight="/home/s2320014/flag_bge_mocheg_finetune/image/BGE-Image-Retrieval-new-20-epoch/BGE_EVA_Token.pth")

        self._text_ids = None
        self._image_ids = None

        self._text_db = None
        self._image_db = None

    def fetch_db_data(self, emb_path, text_db, image_db):
        self._text_ids = np.load(emb_path + "/text_embedding_db_flag_id.npy")
        self._image_ids = np.load(emb_path + "/image_embedding_db_clip_id.npy")
        self._text_db = text_db
        self._image_db = image_db

        print("Loaded evidence DB....")

    def get_evidence_db_ids(self):
        return self._text_ids, self._image_ids
    
    def set_evidence_db_ids(self, text_ids, image_ids):
        self._text_ids = text_ids
        self._image_ids = image_ids

    def retrieve_text_similarity(self, query, candidates_vectors):
        assert len(candidates_vectors) == len(self._text_ids)
        new_candidate_vec = []
        for i in range(0, len(candidates_vectors)):
            if candidates_vectors[i] > 0: 
                new_candidate_vec.append((i, self._text_ids[i]))
        
        sentence_pairs = []
        for sample in new_candidate_vec:
            text_doc = self._text_db.loc[(self._text_db.relevant_document_id == sample[1])]['Origin Document'].values[0]
            sentence_pairs.append([query, text_doc])
        
        # dense_scores = self._text_model.compute_score(sentence_pairs, 
        #                                               max_passage_length=512, 
        #                                               weights_for_different_modes=[0.4, 0.2, 0.4])['dense']

        dense_scores = self._text_model.compute_score(sentence_pairs, normalize=True)
        
        # print(dense_scores)
        # raise Exception
        assert len(dense_scores) == len(new_candidate_vec)
        final_results = []

        for i in range(0, len(dense_scores)):
            final_results.append((new_candidate_vec[i][0], dense_scores[i]))  # (original index ids, score)

        return final_results
    

    def retrieve_image_similarity(self, query, candidates_vectors):
        def find_image_path(image_id, image_db):
            for img in image_db:
                if image_id == img[4]:
                    return img[5]
        assert len(candidates_vectors) == len(self._image_ids)

        new_candidate_vec = []
        for i in range(0, len(candidates_vectors)):
            if candidates_vectors[i] > 0: 
                new_candidate_vec.append((i, find_image_path(self._image_ids[i], self._image_db)))  # original idx, image_id as path
        
        final_results = []
        query_emb = self._viz_model.encode(text=query)
        
        for sample in new_candidate_vec:
            # if sample[1] is not None:
            #     print(sample[1])
            #     raise Exception 
            candi_emb_1 = self._viz_model.encode(image=sample[1])

            # score = query_emb @ candi_emb_1.T
            score = F.cosine_similarity(query_emb, candi_emb_1).detach().cpu().numpy()[0]
            final_results.append((sample[0], score)) # (original index ids, score)
        
        # print(final_results)
        # raise Exception
        return final_results
    
    