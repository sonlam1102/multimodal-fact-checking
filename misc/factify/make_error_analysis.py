import argparse

import pandas as pd
import torch

from read_data import get_dataset

def encode_one_sample(sample):
    claim_id = sample[0]
    claim = sample[1]
    claim_img = sample[2]
    text_evidence = sample[3]
    image_evidence = sample[4]
    label = sample[5]

    label2idx = {
        'Support_Text': 4,
        'Support_Multimodal': 3,
        'Insufficient_Text': 2,
        'Insufficient_Multimodal': 1,
        'Refute': 0,
    }

    label2idx_new = {
        'Support_Text': 2,
        'Support_Multimodal': 2,
        'Insufficient_Text': 1,
        'Insufficient_Multimodal': 1,
        'Refute': 0,
    }

    encoded_sample = {}
    encoded_sample['claim'] = claim
    encoded_sample['claim_id'] = claim_id
    encoded_sample['claim_image'] = claim_img
    encoded_sample["text_evidence"] = text_evidence.tolist()
    encoded_sample["image_evidence"] = image_evidence
    encoded_sample['label'] = label2idx[label]

    return encoded_sample


class ClaimVerificationDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self._data = data

        self._encoded = []
        for d in self._data:
            self._encoded.append(encode_one_sample(d))

    def __len__(self):
        return len(self._encoded)

    def __getitem__(self, idx):
        return self._encoded[idx]

    def to_list(self):
        return self._encoded


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/home/s2320014/data")
    args = parser.parse_args()
    return args


def calculate_data(data):
    claim_ids = []
    claim_labels = []
    image_evidence = []
    text_evidence = []
    for d in data:
        claim_ids.append(d['claim_id'])
        claim_labels.append(d['label'])
        te = [x for x in d['text_evidence'] if str(x) != 'nan']
        ie = 1 if d['image_evidence'] is not None else 0
        ce = 1 if d['claim_image'] is not None else 0
        text_evidence.append(len(te))
        image_evidence.append(ie+ce)

    return pd.DataFrame({
        'claim_id': claim_ids,
        'ground_truth': claim_labels,
        'num_image': image_evidence,
        'num_text': text_evidence
    })


if __name__ == "__main__":
    args = parser_args()

    train, val, test = get_dataset(args.path)
    train_claim = ClaimVerificationDataset(train)
    dev_claim = ClaimVerificationDataset(val)
    test_claim = ClaimVerificationDataset(test)

    data = calculate_data(dev_claim.to_list())
    data.to_csv('dev_analysis.csv', index=False)
