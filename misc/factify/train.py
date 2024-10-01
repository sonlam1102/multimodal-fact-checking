import copy
import math
import random
import datetime
import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn, optim
from transformers import ViTImageProcessor, ViTModel, AutoTokenizer, AutoModel, LongformerTokenizer, LongformerModel, \
    BeitImageProcessor, BeitModel, BigBirdModel, BigBirdTokenizer, DeiTModel, DeiTImageProcessor
import torch.nn.functional as F

from read_data import get_dataset
from tqdm import tqdm, trange


def one_hot(a, num_classes):
    v = np.zeros(num_classes, dtype=int)
    v[a] = 1
    return v


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
    encoded_sample['label'] = torch.tensor(one_hot(label2idx[label], 5), dtype=float)
    encoded_sample['label_new'] = torch.tensor(one_hot(label2idx_new[label], 3), dtype=float)

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

class MultiModalClassification(nn.Module):
    # def vision_model(self):
    #     processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    #     model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    #     model.requires_grad_(False)
    #     return processor, model

    def vision_model(self, type='vit'):
        if type == 'vit':
            processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        if type == 'beit':
            processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
            model = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
        if type == 'deit':
            processor = DeiTImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
            model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")

        model.requires_grad_(False)
        return processor, model

    def text_model(self, pt="roberta-base"):
        processor = AutoTokenizer.from_pretrained(pt)
        model = AutoModel.from_pretrained(pt)
        print(pt)
        return processor, model

    def text_model_long(self, pt="longformer"):
        if pt == 'longformer':
            processor = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
            model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        if pt == 'bigbird':
            model = BigBirdModel.from_pretrained("google/bigbird-roberta-base")
            processor = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")

        model.requires_grad_(False)
        return processor, model

    def __init__(self, device, claim_pt="roberta-base", vision_pt="vit", long_pt='longformer'):
        super(MultiModalClassification, self).__init__()
        self._claim_pt = claim_pt
        self._vision_pt = vision_pt
        self._long_pt = long_pt
        self.text_attention = nn.MultiheadAttention(embed_dim=768, num_heads=2, vdim=768, kdim=768)
        self.image_attention = nn.MultiheadAttention(embed_dim=768, num_heads=2, vdim=768, kdim=768)
        self._text_processor, self._text_model = self.text_model(self._claim_pt)
        self._long_text_processor, self._long_text_model = self.text_model_long(self._long_pt)
        self._image_processor, self._vision_model = self.vision_model(type=self._vision_pt)
        self._device = device

        self.conv = nn.Conv1d(768, 100, stride=1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(768, 100, stride=1, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(768, 100, stride=1, kernel_size=7, padding=3)
        self.fc1 = nn.Linear(768 * 2, 768)

        self.pool = nn.MaxPool1d(2, 2)
        self.softmax = nn.Softmax(dim=1)
        self.leaky_relu = nn.LeakyReLU()

        self.fc_claim = nn.Linear(768, 5)
        self.fc_evidence = nn.Linear(300, 5)

        self.dropout = nn.Dropout(0.2)

        # ablation
        self._fc_evidence_a = nn.Linear(768*2, 5)

    def forward(self, claim_features, label=None):
        device = self._device
        self._vision_model.to(device)
        self._text_model.to(device)
        self._long_text_model.to(device)

        Hc = []
        Ht = []
        Hm = []
        Lb = []

        if label is not None:
            label = label
            for l in label:
                Lb.append(l)

        for claim_feature in claim_features:
            claim = claim_feature['claim']
            text_evidence = claim_feature['text_evidence']
            image_evidence = claim_feature['image_evidence']
            claim_image = claim_feature['claim_image']

            img_set = []
            if image_evidence is not None:
                img_set.append(image_evidence)
                if claim_image is not None:
                    img_set.append(claim_image)
            elif claim_image is not None:
                img_set.append(claim_image)
            else:
                img_set.append(np.zeros((50, 50, 3), np.uint8))

            claim_encoded = self._text_processor(claim, return_tensors="pt", padding=True, truncation=True, max_length=84).to(device)
            # claim_f = self._text_model(**claim_encoded).last_hidden_state.mean(dim=1).to(device)
            claim_f = self._text_model(**claim_encoded).pooler_output.to(device)

            text_encoded = self._long_text_processor(text_evidence, return_tensors="pt", padding=True, truncation=True).to(device)
            # text_feature = self._long_text_model(**text_encoded).last_hidden_state.mean(dim=1).to(device)
            text_feature = self._long_text_model(**text_encoded).pooler_output.to(device)

            try:
                image_encoded = self._image_processor(img_set, return_tensors="pt").to(device)
                # image_feature = self._vision_model(**image_encoded).last_hidden_state.mean(dim=1).to(device)
                image_feature = self._vision_model(**image_encoded).pooler_output.to(device)
            except Exception as e:
                print(image_evidence)
                print(claim_feature)
                raise e

            text_feature = torch.mean(text_feature, 0, keepdim=True)
            image_feature = torch.mean(image_feature, 0, keepdim=True)

            Hc.append(claim_f)
            Ht.append(text_feature)
            Hm.append(image_feature)

        Hc = torch.cat(Hc)
        Ht = torch.cat(Ht)
        Hm = torch.cat(Hm)

        if Lb:
            Lb = torch.stack(Lb)

        text_evidence_features = Ht
        image_evidence_features = Hm

        attention_claim_text, _ = self.text_attention(Hc, text_evidence_features, text_evidence_features)
        attention_claim_img, _ = self.image_attention(Hc, image_evidence_features, image_evidence_features)

        fused_text = self.leaky_relu(self.fc1(torch.cat([attention_claim_text * Hc, attention_claim_text - Hc], 1)))
        fused_img = self.leaky_relu(self.fc1(torch.cat([attention_claim_img * Hc, attention_claim_img - Hc], 1)))

        claim_out = self.fc_claim(Hc)
        claim_out = self.softmax(claim_out)

        # conv modules
        c1_t = F.relu(self.conv(fused_text.T).T)
        c2_t = F.relu(self.conv2(fused_text.T).T)
        c3_t = F.relu(self.conv3(fused_text.T).T)
        conv_t = torch.cat([c1_t, c2_t, c3_t], 1).to(device)

        c1_i = F.relu(self.conv(fused_img.T).T)
        c2_i = F.relu(self.conv2(fused_img.T).T)
        c3_i = F.relu(self.conv2(fused_img.T).T)
        conv_i = torch.cat([c1_i, c2_i, c3_i], 1).to(device)

        combine = torch.cat([conv_t, conv_i], 1).to(device)
        combine = self.pool(combine)
        claim_evidence_out = self.fc_evidence(combine)
        claim_evidence_out = self.softmax(claim_evidence_out)
        # end conv modules

        # # no conv ablation
        # combine = torch.cat([fused_text, fused_img], 1).to(device)
        # claim_evidence_out = self._fc_evidence_a(combine)
        # claim_evidence_out = self.softmax(claim_evidence_out)
        # # end no conv ablation

        # full
        out = torch.mean(torch.stack([claim_out, claim_evidence_out], 0), 0).to(device)

        # no claim
        # out = claim_evidence_out

        return out, Lb


def make_batch(train_data, batch_size=128, shuffle=True):
    claim_ids = []
    claim_labels = []
    claim_features = []

    if shuffle:
        train_data = train_data.to_list() if not isinstance(train_data, list) else train_data
        random.shuffle(train_data.copy())

    for d in train_data:
        claim_ids.append(d['claim_id'])
        claim_labels.append(d['label'])
        claim_features.append(d)

    num_batches = math.ceil(len(train_data) / batch_size)
    train_features_batch = [claim_features[batch_size * y:batch_size * (y + 1)] for y in range(num_batches)]

    train_label_batch = [claim_labels[batch_size * y: batch_size * (y + 1)] for y in range(num_batches)]
    train_id_batch = [claim_ids[batch_size * y:batch_size * (y + 1)] for y in range(num_batches)]

    return train_features_batch, train_label_batch, train_id_batch


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs.squeeze(),  targets.float())
        loss = self.alpha * (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss
        return loss


def train_model(train_data, batch_size, epoch=1, is_val=False, val_data=None, claim_pt="roberta-base", vision_pt="vit", long_pt="longformer", n_gpu=None):
    if n_gpu:
        device = torch.device('cuda:{}'.format(n_gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiModalClassification(device, claim_pt, vision_pt, long_pt)
    model = model.to(device)
    print(model)
    loss_function = FocalLoss(gamma=2)
    loss_function = loss_function.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_vals = []

    print('Training.......')
    best_model = model
    best_acc = 0

    chk_dir = "model_dump/model_verification_{}_{}_{}_{}".format(
        str(claim_pt),
        str(long_pt),
        str(vision_pt),
        str(datetime.datetime.now().strftime("%d-%m_%H-%M"))
    )
    os.makedirs(chk_dir)
    os.makedirs("{}/checkpoint".format(chk_dir))

    X, y, _ = make_batch(train_data, batch_size=batch_size)
    for e in range(epoch):
        model.train()
        total_loss = 0
        print("Epoch {}:\n ".format(e+1))

        for i in trange(len(X)):
            optimizer.zero_grad()
            batch_x = X[i]
            score, lb = model(batch_x, y[i])
            loss = loss_function(score.to(device), lb.to(device))
            loss.backward()

            optimizer.step()
            total_loss = total_loss + loss.item()

        loss_vals.append(total_loss)
        print("Loss: {}\n".format(total_loss))

        if is_val and val_data:
            truelb, predlb, _ = predict(val_data, model, batch_size=batch_size)
            mif1 = f1_score(truelb, predlb, average='micro')
            if mif1 > best_acc:
                best_acc = copy.deepcopy(mif1)
                best_model = copy.deepcopy(model)
            print("Macro F1-score: {}\n".format(f1_score(truelb, predlb, average='macro')))
            print("F1-score: {}\n".format(f1_score(truelb, predlb, average='micro')))
        else:
            best_model = copy.deepcopy(model)
        print('===========\n')

        torch.save({
            'total_epochs': epoch,
            'current_epoch': e,
            'batch_size': batch_size,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, "{}/checkpoint/checkpoint_{}.pt".format(
            str(chk_dir),
            str(e))
        )

    torch.save(best_model, '{}/best_model.pt'.format(chk_dir))

    return best_model, loss_vals, claim_pt


def predict(test_data, model, batch_size, n_gpu=None):
    # if n_gpu:
    #     device = torch.device('cuda:{}'.format(n_gpu) if torch.cuda.is_available() else 'cpu')
    # else:
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = nn.DataParallel(model)
    # model = model.to(device)

    ground_truth = []
    predicts = []
    ids = []

    X, y, z = make_batch(test_data, batch_size=batch_size, shuffle=False)

    model.eval()
    print('Predict.......')

    for i in trange(len(X)):
        batch_x = X[i]
        batch_y = y[i]
        batch_z = z[i]

        scores, lb = model(batch_x)
        scores = scores.reshape(-1, 5)

        if not ids:
            ids = [i for i in batch_z]
        else:
            ids.extend([i for i in batch_z])

        if not ground_truth:
            ground_truth = [np.argmax(label.tolist(), -1) for label in batch_y]
        else:
            ground_truth.extend([np.argmax(label.tolist(), -1) for label in batch_y])

        if not predicts:
            predicts = [np.argmax(score.tolist(), -1) for score in scores]
        else:
            predicts.extend([np.argmax(score.tolist(), -1) for score in scores])

    return ground_truth, predicts, ids


if __name__ == '__main__':
    train, val, test = get_dataset('data')
    dev_claim = ClaimVerificationDataset(val)
    train_claim = ClaimVerificationDataset(train)

    model, loss, _ = train_model(train_claim[70:90], batch_size=5, epoch=5, is_val=True, val_data=dev_claim[1:10])

    gt, prd, ids = predict(dev_claim[1:10], model, 16)
    print("Test result micro: {}\n".format(f1_score(gt, prd, average='micro')))
    print("Test result macro: {}\n".format(f1_score(gt, prd, average='macro')))
    print(confusion_matrix(gt, prd))
