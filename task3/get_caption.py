import argparse
import json
import re

import torch
from PIL import Image
from transformers import pipeline

from read_data import read_text_corpus, read_images_corpus
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

import logging
logging.disable(logging.WARNING)

def retrieve_data_for_verification(train_text, train_images):
    claim_ids = train_text['claim_id'].values
    claim_ids = list(set(claim_ids))

    claim_data = []
    for claim_id in claim_ids:
        df = train_text.loc[(train_text.claim_id == claim_id)]
        text_evidences = df['Evidence'].values
        image_evidences = train_images.loc[(train_images.claim_id == claim_id)]['image'].values
        image_id = train_images.loc[(train_images.claim_id == claim_id)]['image_id'].values
        ruling_statement = df['ruling_outline'].values[0]

        claim_object = (df['Claim'].values[0], text_evidences, image_evidences, df['cleaned_truthfulness'].values[0], claim_id, ruling_statement, image_id)
        claim_data.append(claim_object)

    return claim_data


def get_dataset(path):
    train_text, dev_text, test_text = read_text_corpus(path)
    train_image, dev_image, test_image = read_images_corpus(path)

    val_claim = retrieve_data_for_verification(dev_text, dev_image)
    train_claim = retrieve_data_for_verification(train_text, train_image)
    test_claim = retrieve_data_for_verification(test_text, test_image)

    return train_claim, val_claim, test_claim


def clean_data(text):
    if str(text) == 'nan':
        return ''
    text = text.encode("utf-8", errors='ignore').decode("utf-8")
    text = re.sub("(<p>|</p>|@)+", '', text)
    return str(text.strip())


def get_captioning(sample, device):
    image_evidence = sample[2]
    image_id_list = sample[6]
    claim_id = sample[4]

    encoded_sample = {}
    # encoded_sample['claim_id'] = claim_id
    encoded_sample['claim_id'] = str(claim_id)
    # encoded_sample["image_ids"] = image_id_list
    encoded_sample["image_ids"] = [str(k) for k in image_id_list]

    image_text = []
    if len(image_evidence) > 0:
        image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning",
                                 device=device)
        for t in image_evidence:
            # pill = Image.fromarray(t)
            pill = Image.open(t)
            te = image_to_text(pill)[0]['generated_text']
            image_text.append(te)

    encoded_sample['image_text'] = image_text

    return encoded_sample


def prompt_captioning(sample, device, processor, model):
    claim_text = sample[0]
    # text_evidence = sample[1]
    image_evidence = sample[2]
    image_id_list = sample[6]
    claim_id = sample[4]

    encoded_sample = {}
    # encoded_sample['claim_id'] = claim_id
    encoded_sample['claim_id'] = str(claim_id)
    # encoded_sample["image_ids"] = image_id_list
    encoded_sample["image_ids"] = [str(k) for k in image_id_list]

    image_text = []

    model = model.to(device)

    prompt = "Please describe the detail facts, actions and people of this image according to the given claim: {}".format(claim_text)
    # print("---------------")

    if len(image_evidence) > 0:
        for im in image_evidence:
            # pill = Image.fromarray(im).convert("RGB")
            pill = Image.open(im)
            inputs = processor(images=pill, text=prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=256,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
            )
            # outputs = model.generate(
            #     **inputs,
            #     num_beams=5,
            #     max_new_tokens=256,
            #     min_length=1,
            # )
            generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            print(prompt)
            print("====")
            print(generated_text)
            print("---------------")
            image_text.append(generated_text)

    encoded_sample['image_text'] = image_text
    return encoded_sample


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/home/s2320014/data")
    parser.add_argument('--n_gpu', type=int, default=None)
    args = parser.parse_args()
    return args


def get_caption_dataset(data, device, path, name='train'):
    model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

    encoded = []
    for d in data:
        # encoded.append(get_captioning(d, device))
        encoded.append(prompt_captioning(d, device, processor, model))

    with open(path + '/' + name + '.json', 'w', encoding='utf-8') as f:
        # json.dump(encoded, f, ensure_ascii=False, default=str)
        json.dump(encoded, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    args = parser_args()
    train, val, test = get_dataset(args.path)

    n_gpu = args.n_gpu
    if n_gpu:
        device = torch.device('cuda:{}'.format(n_gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    get_caption_dataset(train, device, args.path, name='train_blip2')
    get_caption_dataset(val, device, args.path, name='dev_blip2')
    get_caption_dataset(test, device, args.path, name='test_blip2')
