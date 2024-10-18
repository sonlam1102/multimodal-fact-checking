from transformers import AutoTokenizer, AutoModelForCausalLM, MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, AutoModelForImageTextToText
import re
import torch
import argparse
from read_data import get_dataset
from PIL import Image
from tqdm import tqdm, trange
import json
from sklearn.metrics import f1_score, accuracy_score

import logging
logging.disable(logging.WARNING)

def clean_data(text):
    if str(text) == 'nan':
        return text
    text = re.sub("(<p>|</p>|@)+", '', text)
    return text.strip()


def encode_one_sample(sample):
    claim = sample[0]
    text_evidence = sample[1]
    image_evidence = sample[2]
    label = sample[3]
    claim_id = sample[4]

    encoded_sample = {}
    encoded_sample["claim_id"] = str(claim_id)
    encoded_sample["claim"] = claim
    encoded_sample["label"] = label
    encoded_sample['text_evidence'] = [clean_data(t) for t in text_evidence]
    encoded_sample['image_evidence'] = image_evidence.tolist()

    return encoded_sample


class ClaimVerificationDataset(torch.utils.data.Dataset):
    def __init__(self, claim_verification_data):
        self._data = claim_verification_data
        # self._processor = processor

        self._encoded = []
        for d in self._data:
            self._encoded.append(encode_one_sample(d))

    def __len__(self):
        return len(self._encoded)

    def __getitem__(self, idx):
        return self._encoded[idx]

    def to_list(self):
        return self._encoded


def load_peft_model_vision(peft_model_name, device="auto"):
    processor = AutoProcessor.from_pretrained(
        peft_model_name,
        model_max_length=2000,
        padding_side="left",
        truncation_side="left",
        token="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",
    )

    quantization_config = BitsAndBytesConfig(
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        load_in_4bit=True,
    )

    model = MllamaForConditionalGeneration.from_pretrained(
        peft_model_name,
        quantization_config=quantization_config,
        token="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",
        device_map="auto",
        use_flash_attention_2=False,
    )

    return processor, model

def load_peft_model_vision2(peft_model_name, device="auto"):
    processor = AutoProcessor.from_pretrained(
        peft_model_name,
        model_max_length=2048,
        padding_side="left",
        truncation_side="left",
        token="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",
        trust_remote_code=True,
    )

    quantization_config = BitsAndBytesConfig(
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        load_in_4bit=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        peft_model_name,
        quantization_config=quantization_config,
        token="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",
        device_map=device,
        # _attn_implementation='eager',
        trust_remote_code=True
    )

    return processor, model


def load_peft_model_text(peft_model_name, device="auto", quantile=True):
    processor = AutoTokenizer.from_pretrained(
        peft_model_name,
        model_max_length=2048,
        padding_side="left",
        truncation_side="left",
        token="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"
    )

    quantization_config = BitsAndBytesConfig(
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        load_in_4bit=True,
    )

    if quantile:
        model = AutoModelForCausalLM.from_pretrained(
        peft_model_name,
        quantization_config=quantization_config,
        token="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",
        device_map=device,
        use_flash_attention_2=True,
    )
    else:
        model = AutoModelForCausalLM.from_pretrained(
        peft_model_name,
        token="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",
        device_map=device,
        use_flash_attention_2=True,
    )

    return processor, model


def make_prompt(text_evidence):
    prompt = f"""
    <|image|><|begin_of_text|>{text_evidence} \n

    Please generate a short paragraph describing the about the consistency of the image based on the given text following this template:
    \n
    <HYPOTHESIS>: Please determining whether the image is consistent with the text or not.
    <EXPLANATION>: Explanation the aligment between the image hypothesis and the text.
    <FINAL ANSWER>: Give one paragraph describing the consistency of the image and text based on the explanation.
    """

    return prompt


@torch.inference_mode()
def do_inference_vision(model, processor, prompt, image):
    image_data = Image.open(image)

    inputs = processor(image_data, prompt, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.2,    
        top_p=0.5,
    )
    return processor.decode(output_ids[0])


@torch.inference_mode()
def do_inference_text(model, processor, prompt):
    inputs = processor(prompt, return_tensors="pt").to(model.device)

    model.generation_config.pad_token_id = processor.pad_token_id
    output_ids = model.generate(
        **inputs,
        max_new_tokens=5,
        do_sample=False,
    )
    return processor.decode(output_ids[0])


@torch.inference_mode()
def do_inference_vision_verification(model, processor, prompt, image):
    prompt = processor.apply_chat_template(prompt, add_generation_prompt=True)
    if len(image) > 0:
        image_data = [Image.open(img) for img in image]
        inputs = processor(images=image_data, text=prompt, padding=True, return_tensors="pt").to(model.device)
    else:
        inputs = processor(text=prompt, padding=True, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        eos_token_id=processor.tokenizer.eos_token_id, 
        max_new_tokens=5,
        do_sample=False,
    )
    # output_ids = output_ids[:, inputs['input_ids'].shape[1]:]
    return processor.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)


def create_align_form(dataset, model, processor, path):
    def get_image_path_new(img_p, path):
        name = img_p.split('/')[-1]
        return path + "/images/" + name
    
    results = []
    print("---performing.....----")
    for sample in tqdm(dataset):
        align_text_image = []
        if len(sample['image_evidence']) > 0:
            for ie in sample['image_evidence']:
                for te in sample['text_evidence']:
                    img = get_image_path_new(ie, path)
                    align_text_image.append({
                        'text': te,
                        'image': ie,
                        'alignment': do_inference_vision(model, processor, make_prompt(te), img)
                    })
        results.append({
            **sample,
            'alignment': align_text_image
        })

    return results


def create_align_form_system(dataset, model, processor, path):
    def get_image_path_new(img_p, path):
        name = img_p.split('/')[-1]
        return path + "/images/" + name
    
    results = []
    print("---performing.....----")
    for sample in tqdm(dataset):
        align_text_image = []
        if len(sample['image_evidence']) > 0:
            for ie in sample['image_evidence']:
                for te in sample['text_evidence']:
                    image = get_image_path_new([*ie.values()][0], path)
                    align_text_image.append({
                        'text': [*te.values()][0],
                        'image': image,
                        'alignment': do_inference_vision(model, processor, make_prompt([*te.values()][0]), image)
                    })
        results.append({
            **sample,
            'alignment': align_text_image
        })

    return results


def read_augmented_data(data):
    for d in data:
        if len(d['alignment']) > 0:
            for a in d['alignment']:
                temp = a['alignment'].split("\n\n\n\n")[-1]
                a['clean_alignment'] = temp.replace('<|eot_id|>', '').strip()
    return data

def retrieve_verification_results(data):
    def filter_results(response):
        response = response.split("<RESPONSE>")[-1]
        if "supported" in response:
            return "supported"
        elif "refuted" in response:
            return "refuted"
        else:
            return "NEI"

    label2inx = {
        "supported": 2,
        "NEI": 1,
        "refuted": 0
    }

    ground_truth = []
    predict = []
    for d in data:
        predict.append(label2inx[filter_results(d['results'])])
        ground_truth.append(label2inx[d['label']])
    
    return ground_truth, predict


def make_verification_prompt(claim, text_evidence, image_guides, path):
    def make_image_description_evidence(image_explaination):
        expl = image_explaination.split("<EXPLANATION>")[-1]
        expl = expl.replace("<FINAL ANSWER>:", "")
        expl = expl.replace("\n", "")

        return expl

    if len(image_guides) > 0:
        img_guides = ""
        eidx = 1
        for im in image_guides:
            img_guides += f"""
            Evidence {eidx}: \n
            Text evidence: {im['text']} \n
            Image evidence {make_image_description_evidence(im['clean_alignment'])} \n
            """
            eidx += 1
        
        prompt = f"""
        You are an assistant to perform checking the truthfulness of a claim. 
        The claim is: {claim} \n
        Here are collected evidence about the claim. 
        Please consulting these evidences for verifying the truthfulness of the claim: \n
        {img_guides}
        
        Based on those given clues, please determining the truthfulness of the claim. The results must be one of these three values: refuted, supported or not enoght information.
        <RESPONSE>: 
        """
        # print(prompt)
        # raise Exception
    else:
        evidences = ""
        eidx = 1
        for t in text_evidence:
            evidences += f"""
            Evidence {eidx}: {t} \n
            """
            eidx += 1
        prompt = f"""
        You are an assistant to perform checking the truthfulness of a claim. 
        The claim is: {claim} \n
        Please consulting these evidences for verifying the claim: 
            {evidences} \n
        
        Based on those given clues, please determining the truthfulness of the claim. The results must be one of these three values: refuted, supported or not enoght information.
        <RESPONSE>:
        """
        # print(prompt)
        # raise Exception
    return prompt


def make_verification_prompt_vision(claim, text_evidence, image_guides, path):
    def parse_image_path(img_evidence):
        return path + "/images/" + img_evidence.split("/")[-1]

    images_list = []
    if len(image_guides) > 0:
        content = [
            {"type": "text", "text": "You are an assistant to perform checking the truthfulness of a claim."},
            {"type": "text", "text": "The claim is: {}.".format(claim)},
            {"type": "text", "text": "Please consulting these evidences for verifying the claim: "},
        ]
        for im in image_guides:
            content.append({"type": "text", "text": im['text']})
            content.append({"type": "image"})
            content.append({"type": "text", "text": im['alignment']})
            
            images_list.append(parse_image_path(im['image']))
    else:
        content = [
            {"type": "text", "text": "You are an assistant to perform checking the truthfulness of a claim."},
            {"type": "text", "text": "The claim is: {}.".format(claim)},
            {"type": "text", "text": "Please consulting these evidences for verifying the claim: "},
        ]
        for t in text_evidence:
            content.append({"type": "text", "text": t})
    
    content.append(
        {"type": "text", "text": "Based on those given clues, please determining the truthfulness of the claim. The results must be one of these three values: refuted, supported or not enoght information. \n<RESPONSE>:"},
    )
    prompt = [
        {
            "role": "user",
            "content": content
        }
    ]
    return prompt, images_list

def create_verification_prompt(dataset, model, processor, path):
    results = []
    new_dataset = read_augmented_data(dataset)
    print("---performing verification .....----")
    for sample in tqdm(new_dataset):
        prompt = make_verification_prompt(sample['claim'], sample['text_evidence'], sample['alignment'], path)
        try:
            results.append({
                **sample,
                'results': do_inference_text(model, processor, prompt)
            })
        except Exception as e:
            print(e)
            print(sample['claim_id'])
            results.append({
                **sample,
                'results': "This claim is supported"
            })
    return results


def create_verification_prompt_vision(dataset, model, processor, path):
    results = []
    new_dataset = read_augmented_data(dataset)
    print("---performing verification .....----")
    for sample in tqdm(new_dataset):
        prompt, lst_image = make_verification_prompt_vision(sample['claim'], sample['text_evidence'], sample['alignment'], path)
        try:
            results.append({
                **sample,
                'results': do_inference_vision_verification(model, processor, prompt, lst_image)
            })
        except Exception as e:
            print(e)
            print(sample['claim_id'])
            results.append({
                **sample,
                'results': "This claim is supported"
            })
    return results

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/home/s2320014/data")
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--system', default=False, action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser_args()
    # processor, model = load_peft_model_text("meta-llama/Llama-3.1-70B-Instruct")
    # processor, model = load_peft_model_text("Qwen/Qwen2.5-32B-Instruct")
    # processor, model = load_peft_model_text("Qwen/Qwen2.5-72B-Instruct")
    processor, model = load_peft_model_vision2("llava-hf/llava-v1.6-vicuna-7b-hf")

    # # Demo 
    # train, val, test = get_dataset(args.path)
    # dev_claim = ClaimVerificationDataset(val)
    # test_claim = ClaimVerificationDataset(test)
    # sample = dev_claim[0]
    # sample_prompt = make_prompt(sample['text_evidence'][0])
    # print(sample['image_evidence'][0])
    # output = do_inference_vision(model, processor, sample_prompt, sample['image_evidence'][0])
    # print(output)
    
    # # Prompting with LLMs 
    with open("./mocheg_claim_llama3.2_test.json", "r") as f:
        dataset = json.load(f)
    f.close()
    
    # results = create_verification_prompt(dataset, model, processor, args.path)
    results = create_verification_prompt_vision(dataset, model, processor, args.path)
    with open('./mocheg_verification_test_llava-next.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    f.close()
    

    # # testing 
    # with open("./sample_dump/pred_verification_Qwen2.5_test.json", "r") as f:
    #     results = json.load(f)
    # f.close()

    g, p = retrieve_verification_results(results)
    print("Test result micro: {}\n".format(f1_score(g, p, average='micro')))
    print("Test result macro: {}\n".format(f1_score(g, p, average='macro')))
    print("Test result Accuracy: {}\n".format(accuracy_score(g, p)))
