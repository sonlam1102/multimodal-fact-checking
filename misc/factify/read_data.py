import glob

import cv2
import pandas as pd
from tqdm import tqdm, trange

DATA_PATH = 'data'


def read_data(path):
    train = pd.read_csv(path+"/public_folder/train.csv")
    dev = pd.read_csv(path+"/public_folder/val.csv")
    test = pd.read_csv(path+"/public_folder/test_gold.csv")

    return train, dev, test


def read_images(path):
    document_f = []
    ext = ['jpg', 'jpeg']
    for e in ext:
        for img in glob.glob(path + "/*." + e):
            document_f.append(img)
    img_object = []

    for i in trange(len(document_f)):
        p = document_f[i]
        name_img = p.split('/')[-1].split('.')[0]
        id_img = int(name_img.split("_")[0])

        if len(name_img.split("_")) > 1: #is claim
            is_claim = True
            img_value = cv2.imread(p, cv2.IMREAD_COLOR)
        else:
            is_claim = False
            img_value = cv2.imread(p, cv2.IMREAD_COLOR)

        img_object.append((id_img, img_value, is_claim))

    return img_object


def get_image_corpus(path):
    train = path + "/images_set/train"
    dev = path + "/images_set/val"
    test = path + "/images_set/test"

    train_img = read_images(train)
    dev_img = read_images(dev)
    test_img = read_images(test)

    return train_img, dev_img, test_img


def retrieve_data_for_verification(text, images):
    claim_ids = text['Id'].values
    claim_ids = list(set(claim_ids))
    claim_data = []

    for claim_id in claim_ids:
        df = text.loc[(text.Id == claim_id)]

        claim = df['claim'].values[0]
        text_evidences = df['document'].values

        claim_img = None
        image_evidence = None

        for img in images:
            img_id = img[0]
            img_value = img[1]
            is_img_claim = img[2]
            if img_id == claim_id and is_img_claim:
                claim_img = img_value

        for img in images:
            img_id = img[0]
            img_value = img[1]
            is_img_claim = img[2]
            if img_id == claim_id and not is_img_claim:
                image_evidence = img_value

        label = df['Category'].values[0]

        claim_object = (claim_id, claim, claim_img, text_evidences, image_evidence, label)
        claim_data.append(claim_object)

    return claim_data


def get_dataset(PATH):
    train_text, dev_text, test_text = read_data(PATH)
    train_image, dev_image, test_image = get_image_corpus(PATH)

    train = retrieve_data_for_verification(train_text, train_image)
    dev = retrieve_data_for_verification(dev_text, dev_image)
    test = retrieve_data_for_verification(test_text, test_image)

    return train, dev, test


if __name__ == '__main__':
    train, dev, test = get_dataset(DATA_PATH)
    # img = read_img_from_url("https://i0.wp.com/www.altnews.in/wp-content/uploads/2020/05/2016_%E0%A4%95%E0%A5%80_%E0%A4%9B%E0%A5%87%E0%A5%9C%E0%A4%96%E0%A4%BE%E0%A4%A8%E0%A5%80_%E0%A4%95%E0%A5%87_%E0%A4%AC%E0%A4%BE%E0%A4%A6_%E0%A4%AE%E0%A4%B9%E0%A4%BF%E0%A4%B2%E0%A4%BE_%E0%A4%95%E0%A5%80_%E0%A4%AA%E0%A4%BF%E0%A4%9F%E0%A4%BE%E0%A4%88_%E0%A4%95%E0%A4%B0%E0%A4%A8%E0%A5%87_%E0%A4%95%E0%A5%80_%E0%A4%A4%E0%A4%B8%E0%A5%8D%E0%A4%B5%E0%A5%80%E0%A4%B0%E0%A5%87%E0%A4%82_%E0%A4%A4%E0%A4%AC_%E0%A4%B8%E0%A5%87_%E0%A4%B9%E0%A5%8B_%E0%A4%B0%E0%A4%B9%E0%A5%80_%E0%A4%B0%E0%A4%B9%E0%A5%80_%E0%A4%B5%E0%A4%BE%E0%A4%AF%E0%A4%B0%E0%A4%B2_Alt_News.png?resize=419,542")
    pass
