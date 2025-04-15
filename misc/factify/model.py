import cv2

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn, optim
from transformers import ViTImageProcessor, ViTModel, AutoTokenizer, AutoModel, LongformerTokenizer, LongformerModel, \
    BeitImageProcessor, BeitModel, BigBirdModel, BigBirdTokenizer, DeiTModel, DeiTImageProcessor
import torch.nn.functional as F

from read_data import get_dataset
from tqdm import tqdm, trange
from PIL import Image


class MultiModalClassification(nn.Module):
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
            # image_evidence = claim_feature['image_evidence']
            # claim_image = claim_feature['claim_image']

            # image_evidence = np.array(Image.open(claim_feature['image_evidence'])) if claim_feature['image_evidence'] is not None else None
            # claim_image = np.array(Image.open(claim_feature['claim_image'])) if claim_feature['claim_image'] is not None else None
            try:
                image_evidence = cv2.imread(claim_feature['image_evidence'], cv2.IMREAD_COLOR)
            except Exception as e:
                image_evidence = None

            try:
                claim_image = cv2.imread(claim_feature['claim_image'], cv2.IMREAD_COLOR)
            except Exception as e:
                claim_image = None

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
            text_encoded = self._long_text_processor(text_evidence, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
            # text_feature = self._long_text_model(**text_encoded).last_hidden_state.mean(dim=1).to(device)
            text_feature = self._long_text_model(**text_encoded).pooler_output.to(device)

            try:
                image_encoded = self._image_processor(img_set, return_tensors="pt").to(device)
                # image_feature = self._vision_model(**image_encoded).last_hidden_state.mean(dim=1).to(device)
                image_feature = self._vision_model(**image_encoded).pooler_output.to(device)
            except Exception as e:
                print(img_set)
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
        
        # Attention modules
        attention_claim_text, _ = self.text_attention(Hc, text_evidence_features, text_evidence_features)
        attention_claim_img, _ = self.image_attention(Hc, image_evidence_features, image_evidence_features)

        fused_text = self.leaky_relu(self.fc1(torch.cat([attention_claim_text * Hc, attention_claim_text - Hc], 1)))
        fused_img = self.leaky_relu(self.fc1(torch.cat([attention_claim_img * Hc, attention_claim_img - Hc], 1)))
        # End attention module
        
        # # No attention
        # fused_text = self.leaky_relu(self.fc1(torch.cat([text_evidence_features * Hc, text_evidence_features - Hc], 1)))
        # fused_img = self.leaky_relu(self.fc1(torch.cat([image_evidence_features * Hc, image_evidence_features - Hc], 1)))

        # # end no attention

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
    

### ABLATION Study
## No Claim
class MultiModalClassificationNoClaim(nn.Module):
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
        super(MultiModalClassificationNoClaim, self).__init__()
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
            # image_evidence = claim_feature['image_evidence']
            # claim_image = claim_feature['claim_image']

            # image_evidence = np.array(Image.open(claim_feature['image_evidence'])) if claim_feature['image_evidence'] is not None else None
            # claim_image = np.array(Image.open(claim_feature['claim_image'])) if claim_feature['claim_image'] is not None else None
            try:
                image_evidence = cv2.imread(claim_feature['image_evidence'], cv2.IMREAD_COLOR)
            except Exception as e:
                image_evidence = None

            try:
                claim_image = cv2.imread(claim_feature['claim_image'], cv2.IMREAD_COLOR)
            except Exception as e:
                claim_image = None

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
            text_encoded = self._long_text_processor(text_evidence, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
            # text_feature = self._long_text_model(**text_encoded).last_hidden_state.mean(dim=1).to(device)
            text_feature = self._long_text_model(**text_encoded).pooler_output.to(device)

            try:
                image_encoded = self._image_processor(img_set, return_tensors="pt").to(device)
                # image_feature = self._vision_model(**image_encoded).last_hidden_state.mean(dim=1).to(device)
                image_feature = self._vision_model(**image_encoded).pooler_output.to(device)
            except Exception as e:
                print(img_set)
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
        
        # Attention modules
        attention_claim_text, _ = self.text_attention(Hc, text_evidence_features, text_evidence_features)
        attention_claim_img, _ = self.image_attention(Hc, image_evidence_features, image_evidence_features)

        fused_text = self.leaky_relu(self.fc1(torch.cat([attention_claim_text * Hc, attention_claim_text - Hc], 1)))
        fused_img = self.leaky_relu(self.fc1(torch.cat([attention_claim_img * Hc, attention_claim_img - Hc], 1)))
        # End attention module

        # claim_out = self.fc_claim(Hc)
        # claim_out = self.softmax(claim_out)

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

        # no claim
        out = claim_evidence_out

        return out, Lb


## No CONV
class MultiModalClassificationNoCONV(nn.Module):
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
        super(MultiModalClassificationNoCONV, self).__init__()
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

            try:
                image_evidence = cv2.imread(claim_feature['image_evidence'], cv2.IMREAD_COLOR)
            except Exception as e:
                image_evidence = None

            try:
                claim_image = cv2.imread(claim_feature['claim_image'], cv2.IMREAD_COLOR)
            except Exception as e:
                claim_image = None

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
            text_encoded = self._long_text_processor(text_evidence, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
            # text_feature = self._long_text_model(**text_encoded).last_hidden_state.mean(dim=1).to(device)
            text_feature = self._long_text_model(**text_encoded).pooler_output.to(device)

            try:
                image_encoded = self._image_processor(img_set, return_tensors="pt").to(device)
                # image_feature = self._vision_model(**image_encoded).last_hidden_state.mean(dim=1).to(device)
                image_feature = self._vision_model(**image_encoded).pooler_output.to(device)
            except Exception as e:
                print(img_set)
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
        
        # Attention modules
        attention_claim_text, _ = self.text_attention(Hc, text_evidence_features, text_evidence_features)
        attention_claim_img, _ = self.image_attention(Hc, image_evidence_features, image_evidence_features)

        fused_text = self.leaky_relu(self.fc1(torch.cat([attention_claim_text * Hc, attention_claim_text - Hc], 1)))
        fused_img = self.leaky_relu(self.fc1(torch.cat([attention_claim_img * Hc, attention_claim_img - Hc], 1)))
        # End attention module
        
        claim_out = self.fc_claim(Hc)
        claim_out = self.softmax(claim_out)

        # no conv ablation
        combine = torch.cat([fused_text, fused_img], 1).to(device)
        claim_evidence_out = self._fc_evidence_a(combine)
        claim_evidence_out = self.softmax(claim_evidence_out)
        # end no conv ablation

        # full
        out = torch.mean(torch.stack([claim_out, claim_evidence_out], 0), 0).to(device)

        return out, Lb
    
## No Attention
class MultiModalClassificationNoAttention(nn.Module):
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
        super(MultiModalClassificationNoAttention, self).__init__()
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

            try:
                image_evidence = cv2.imread(claim_feature['image_evidence'], cv2.IMREAD_COLOR)
            except Exception as e:
                image_evidence = None

            try:
                claim_image = cv2.imread(claim_feature['claim_image'], cv2.IMREAD_COLOR)
            except Exception as e:
                claim_image = None

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
            text_encoded = self._long_text_processor(text_evidence, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
            # text_feature = self._long_text_model(**text_encoded).last_hidden_state.mean(dim=1).to(device)
            text_feature = self._long_text_model(**text_encoded).pooler_output.to(device)

            try:
                image_encoded = self._image_processor(img_set, return_tensors="pt").to(device)
                # image_feature = self._vision_model(**image_encoded).last_hidden_state.mean(dim=1).to(device)
                image_feature = self._vision_model(**image_encoded).pooler_output.to(device)
            except Exception as e:
                print(img_set)
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
        
        # No attention
        fused_text = self.leaky_relu(self.fc1(torch.cat([text_evidence_features * Hc, text_evidence_features - Hc], 1)))
        fused_img = self.leaky_relu(self.fc1(torch.cat([image_evidence_features * Hc, image_evidence_features - Hc], 1)))

        # end no attention

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

        # full
        out = torch.mean(torch.stack([claim_out, claim_evidence_out], 0), 0).to(device)

        return out, Lb


## No fusion
class MultiModalClassificationNoFusion(nn.Module):
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
        super(MultiModalClassificationNoFusion, self).__init__()
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
            
            try:
                image_evidence = cv2.imread(claim_feature['image_evidence'], cv2.IMREAD_COLOR)
            except Exception as e:
                image_evidence = None

            try:
                claim_image = cv2.imread(claim_feature['claim_image'], cv2.IMREAD_COLOR)
            except Exception as e:
                claim_image = None

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
            text_encoded = self._long_text_processor(text_evidence, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
            # text_feature = self._long_text_model(**text_encoded).last_hidden_state.mean(dim=1).to(device)
            text_feature = self._long_text_model(**text_encoded).pooler_output.to(device)

            try:
                image_encoded = self._image_processor(img_set, return_tensors="pt").to(device)
                # image_feature = self._vision_model(**image_encoded).last_hidden_state.mean(dim=1).to(device)
                image_feature = self._vision_model(**image_encoded).pooler_output.to(device)
            except Exception as e:
                print(img_set)
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
        
        # Attention modules
        attention_claim_text, _ = self.text_attention(Hc, text_evidence_features, text_evidence_features)
        attention_claim_img, _ = self.image_attention(Hc, image_evidence_features, image_evidence_features)

        claim_out = self.fc_claim(Hc)
        claim_out = self.softmax(claim_out)

        # conv modules
        c1_t = F.relu(self.conv(attention_claim_text.T).T)
        c2_t = F.relu(self.conv2(attention_claim_text.T).T)
        c3_t = F.relu(self.conv3(attention_claim_text.T).T)
        conv_t = torch.cat([c1_t, c2_t, c3_t], 1).to(device)

        c1_i = F.relu(self.conv(attention_claim_img.T).T)
        c2_i = F.relu(self.conv2(attention_claim_img.T).T)
        c3_i = F.relu(self.conv2(attention_claim_img.T).T)
        conv_i = torch.cat([c1_i, c2_i, c3_i], 1).to(device)

        combine = torch.cat([conv_t, conv_i], 1).to(device)
        combine = self.pool(combine)
        claim_evidence_out = self.fc_evidence(combine)
        claim_evidence_out = self.softmax(claim_evidence_out)
        # end conv modules

        # full
        out = torch.mean(torch.stack([claim_out, claim_evidence_out], 0), 0).to(device)

        return out, Lb
