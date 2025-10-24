from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, BartForConditionalGeneration, BartTokenizer
from copy import deepcopy
import torch 
import json
from tqdm import tqdm

def do_sum(claim, text_evidence, model, tokenizer, device):
    command = "summarize: {} </s> {} </s>".format(claim, text_evidence)
    model = model.to(device)
    input = tokenizer(command, return_tensors="pt", padding="max_length", max_length=1024, truncation=True, return_attention_mask=True)
    out_results = model.generate(
            input_ids=input.input_ids.to(device),
            attention_mask=input.attention_mask.to(device),
            num_beams=3,
            max_length=500,
            min_length=0,
            no_repeat_ngram_size=2,
        )

    preds = [tokenizer.decode(p, skip_special_tokens=True) for p in out_results]

    return preds[0]


def make_summary(data, model, tokenizer, device):
    new_result = []
    for d in tqdm(data):
        new_evidence = []
        claim = d['claim']
        for ev in d['text_evidence']:
            key = list(ev.keys())[0]
            val = list(ev.values())[0]

            new_evidence.append({
                key: do_sum(claim, val, model, tokenizer, device)
            })

        d['original_text_evidence'] = deepcopy(d['text_evidence'])
        d['text_evidence'] = new_evidence

        new_result.append(d)
    
    return new_result


def make_summary_bge(data, model, tokenizer, device):
    new_result = []
    for d in tqdm(data):
        claim = d['query']
        
        pos = deepcopy(d["pos"])
        neg = deepcopy(d["neg"])
        
        if len(pos) == 0 or len(neg) == 0:
            continue
        
        new_pos = []
        new_neg = []
        for kp in pos:
            new_pos.append(do_sum(claim, kp, model, tokenizer, device))
        
        for kn in neg:
            new_neg.append(do_sum(claim, kn, model, tokenizer, device))
        
        d["pos"] = new_pos
        d['neg'] = new_neg
        new_result.append(d)
    
    return new_result


if __name__ == '__main__':
    model = BartForConditionalGeneration.from_pretrained("./model/summary/facebook/bart-base/model")
    tokenizer = BartTokenizer.from_pretrained("./model/summary/facebook/bart-base/tokenizer")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # with open("./sample_dump/pred_retrieval_dev.json", "r") as f:
    #     dev_claim_system = json.load(f)
    # f.close()

    # with open("./sample_dump/pred_retrieval_test.json", "r") as f:
    #     test_claim_system = json.load(f)
    # f.close()


    # new_sum_pred_dev = make_summary(dev_claim_system, model, tokenizer, device)
    # new_sum_pred_test = make_summary(test_claim_system, model, tokenizer, device)

    # with open('./sample_dump/pred_retrieval_dev_summarized.json', 'w', encoding='utf-8') as f:
    #     json.dump(new_sum_pred_dev, f, ensure_ascii=False, indent=4)
    # f.close()


    # with open('./sample_dump/pred_retrieval_test_summarized.json', 'w', encoding='utf-8') as f:
    #     json.dump(new_sum_pred_test, f, ensure_ascii=False, indent=4)
    # f.close()
    
    
    with open('./train_text_candidates.jsonl', "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    f.close()
    
    new_sum = make_summary_bge(data, model, tokenizer, device)
    
    with open('./train_text_candidates_sum.jsonl', 'w', encoding="utf-8") as outfile:
        for entry in new_sum:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')
    outfile.close()
