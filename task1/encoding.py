import argparse

import torch
from FlagEmbedding import BGEM3FlagModel
import numpy as np

from visualized_bge import Visualized_BGE
# from FlagEmbedding.visual.modeling import Visualized_BGE

from transformers import ViltImageProcessor
from torch.utils.data import DataLoader
from accelerate import Accelerator
import torch.nn as nn
import clip
from PIL import Image

from read_data import get_text_evidences_db, get_image_evidences_db_path_only, get_text_evidences_sentence_db, get_image_evidences_db
from tqdm import tqdm, trange

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, default="../data")
    parser.add_argument('--n_gpu', type=int, default=None)
    args = parser.parse_args()
    return args


def encoding_text(model, textdb):
    lst_text_db = textdb['Origin Document'].values.tolist()
    lst_text_id = textdb['relevant_document_id'].values.tolist()
    loaders = torch.utils.data.DataLoader(lst_text_db, batch_size=4, shuffle=False)
    embedding = None
    embedding_id = np.array(lst_text_id)
    for d in tqdm(loaders):
        if embedding is None:
            embedding = model.encode(d)['dense_vecs']
        else:
            embedding = np.append(embedding, model.encode(d)['dense_vecs'], axis=0)

    print(embedding.shape)
    print(embedding_id.shape)
    np.save("./encode_embedding/text_embedding_db_flag_finetune.npy", embedding)
    np.save("./encode_embedding/text_embedding_db_flag_finetune_id.npy", embedding_id)


def encoding_sentence(model, sentenceDB):
    lst_sentence_db = sentenceDB['paragraph'].values.tolist()
    lst_sentence_ids = sentenceDB['2903-15073-0'].values.tolist()
    loaders = torch.utils.data.DataLoader(lst_sentence_db, batch_size=4, shuffle=False)

    embedding = None
    embedding_id = np.array(lst_sentence_ids)
    for d in tqdm(loaders):
        if embedding is None:
            embedding = model.encode(d)['dense_vecs']
        else:
            embedding = np.append(embedding, model.encode(d)['dense_vecs'], axis=0)

    print(embedding.shape)
    print(embedding_id.shape)
    np.save("./encode_embedding/sentence_embedding_db_flag.npy", embedding)
    np.save("./encode_embedding/sentence_embedding_db_flag_id.npy", embedding_id)


def encoding_image(viz_model, imagedb):
    list_images = []
    lst_ids = []
    for im in imagedb:
        list_images.append(im[5])
        lst_ids.append(im[4])

    embedding = None
    embedding_id = np.array(lst_ids)

    viz_model.eval()
    print("--Encoding by Visualized BGE--")
    for d in tqdm(list_images):
        if embedding is None:
            embedding = viz_model.encode(image=d).cpu().detach().numpy()
        else:
            embedding = np.append(embedding, viz_model.encode(image=d).cpu().detach().numpy(), axis=0)


    print(embedding.shape)
    print(embedding_id.shape)
    np.save("./encode_embedding/image_embedding_db_flag_finetune.npy", embedding)
    np.save("./encode_embedding/image_embedding_db_flag_finetune_id.npy", embedding_id)


def encoding_image_with_text(viz_model, imagedb, textdb):
    list_images = []
    lst_ids = []
    lst_text = []
    for im in imagedb:
        list_images.append(im[5])
        lst_ids.append(im[4])
        try:
            text_img = textdb.loc[(textdb.relevant_document_id == im[1])]['Origin Document'].values
            if len(text_img) > 0:
                lst_text.append(text_img[0])
            else:
                lst_text.append("")
        except Exception as errr:
            # print(errr)
            # raise errr
            pass

    embedding = None
    embedding_id = np.array(lst_ids)

    assert len(lst_text) == len(list_images)
    viz_model.eval()
    print("--Encoding by Visualized BGE--")

    error = 0
    for i in trange(0, len(list_images)):
        # with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        #     if embedding is None:
        #         try:
        #             embedding = viz_model.encode(image=list_images[i], text=lst_text[i]).cpu().detach().numpy()
        #         except Exception as errr:
        #             embedding = viz_model.encode(image=list_images[i]).cpu().detach().numpy()
        #             print(errr)

        #     else:
                # try:
                #     embedding = np.append(embedding, viz_model.encode(image=list_images[i], text=lst_text[i]).cpu().detach().numpy(), axis=0)
                # except Exception as errr:
                #     embedding = np.append(embedding, viz_model.encode(image=list_images[i]).cpu().detach().numpy(), axis=0)
                #     print(errr)
        # try:
        #     if embedding is None:
        #         embedding = viz_model.encode(image=list_images[i], text=lst_text[i]).cpu().detach().numpy()
        #     else:
        #         embedding = np.append(embedding, viz_model.encode(image=list_images[i], text=lst_text[i]).cpu().detach().numpy(), axis=0)
        # except Exception as errrr:
        #     print(lst_text[i])
        #     raise errrr
        #     error = error + 1
        #     if embedding is None:
        #         embedding = viz_model.encode(image=list_images[i]).cpu().detach().numpy()
        #     else:
        #         embedding = np.append(embedding, viz_model.encode(image=list_images[i]).cpu().detach().numpy(), axis=0)
        if embedding is None:
            embedding = viz_model.encode(image=list_images[i], text=lst_text[i]).cpu().detach().numpy()
        else:
            embedding = np.append(embedding, viz_model.encode(image=list_images[i], text=lst_text[i]).cpu().detach().numpy(), axis=0)

    print(embedding.shape)
    print(embedding_id.shape)
    np.save("./encode_embedding/image_embedding_db_flag_2.npy", embedding)
    np.save("./encode_embedding/image_embedding_db_flag_id_2.npy", embedding_id)
    

def encoding_image2(viz_model, imagedb, device):
    list_images = []
    lst_ids = []
    for im in imagedb:
        list_images.append(im[5])
        lst_ids.append(im[4])

    embedding_id = np.array(lst_ids)
    # img_loader = DataLoader(list_images, batch_size=32, shuffle=False)

    print("--Encoding by ViLT--")
    embedding = None
    for b in tqdm(list_images):
        # try:
        #     image_encoded = viz_model(b, return_tensors="pt")
        # except Exception as e:
        #     print(e)
        #     image_encoded = viz_model(np.zeros((50, 50, 3), np.uint8), return_tensors="pt")

        # embedding.append(image_encoded)
        if embedding is None:
            embedding = viz_model(b, return_tensors="np")
        else:
            embedding = np.append(embedding, viz_model(b, return_tensors="np"), axis=0)

    # embedding = torch.stack(embedding)

    # embedding = embedding.cpu().detach().numpy()
    # image_encoded = image_encoded.detach().cpu()

    print(embedding.shape)
    print(embedding_id.shape)
    np.save("./encode_embedding/image_embedding_db_vilt.npy", embedding)
    np.save("./encode_embedding/image_embedding_db_vilt_id.npy", embedding_id)


def encoding_image3(image_model, processor, imagedb, device):
    list_images = []
    lst_ids = []
    for im in imagedb:
        list_images.append(im[5])
        lst_ids.append(im[4])

    embedding_id = np.array(lst_ids)
    # img_loader = DataLoader(list_images, batch_size=32, shuffle=False)

    print("--Encoding by CLIP--")
    embedding = None
    for img in tqdm(list_images):
        im_process = processor(Image.open(img)).unsqueeze(0).to(device)
        image_features = image_model.encode_image(im_process)
        if embedding is None:
            embedding = image_features.cpu().detach().numpy()
        else:
            embedding = np.append(embedding, image_features.cpu().detach().numpy(), axis=0)

    print(embedding.shape)
    print(embedding_id.shape)
    np.save("./encode_embedding/image_embedding_db_clip.npy", embedding)
    np.save("./encode_embedding/image_embedding_db_clip_id.npy", embedding_id)


if __name__ == '__main__':
    args = parser_args()
    if args.n_gpu:
        device = torch.device("cuda:{}".format(args.n_gpu) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_PATH = args.db_path
    # Original Models
    # model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False, device=device)
    # viz_model = Visualized_BGE(model_name_bge="BAAI/bge-visualized-base-en-v1.5", model_weight="./Visualized_m3.pth")
    # image_model, processor = clip.load("ViT-L/14@336px", device=device)

    # Fine-tune model
    # model = BGEM3FlagModel('/home/s2320014/flag_bge_mocheg_finetune/text/BGE-Text-Retrieval', use_fp16=False, device=device)
    viz_model = Visualized_BGE(model_name_bge="/home/s2320014/flag_bge_mocheg_finetune/image/BGE-Image-Retrieval-new-50-epoch/checkpoint-70000", model_weight="/home/s2320014/flag_bge_mocheg_finetune/image/BGE-Image-Retrieval-new-50-epoch/checkpoint-70000/BGE_EVA_Token.pth")
    
    # viz_model = nn.DataParallel(viz_model, device_ids = [0, 1])
    # viz_model = Visualized_BGE(model_name_bge="BAAI/bge-base-en-v1.5", model_weight="./Visualized_base_en_v1.5.pth")
    # viz_model = ViltImageProcessor.from_pretrained("dandelin/vilt-b32-finetuned-coco")
    # viz_model = Visualized_BGE(model_name_bge="BAAI/bge-base-en-v1.5", model_weight="./Visualized_base_en_v1.5.pth")

    # text_evidences = get_text_evidences_db(DATA_PATH)
    # print(len(text_evidences))
    # encoding_text(model, text_evidences)

    image_evidences = get_image_evidences_db_path_only(DATA_PATH)
    print(len(image_evidences))
    # encoding_image_with_text(viz_model, image_evidences, text_evidences)
    encoding_image(viz_model, image_evidences)
    # encoding_image3(image_model, processor, image_evidences, device)

    # image_evidences = get_image_evidences_db(DATA_PATH)
    # print(len(image_evidences))
    # encoding_image2(viz_model, image_evidences, device)

    # sentence_evidences = get_text_evidences_sentence_db(DATA_PATH)
    # print(len(sentence_evidences))
    # encoding_sentence(model, sentence_evidences)
