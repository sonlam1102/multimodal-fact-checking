import pandas as pd
import cv2
import glob
from PIL import Image
from tqdm import tqdm, trange
import numpy as np
import hickle as hkl
import h5py

DATA_PATH = '../data'

def read_text_corpus(path):
    train = path + '/train/Corpus2.csv'
    dev = path + '/val/Corpus2.csv'
    test = path + '/test/Corpus2.csv'

    train_data = pd.read_csv(train, low_memory=False)
    val_data = pd.read_csv(dev, low_memory=False)
    test_data = pd.read_csv(test, low_memory=False)

    return (train_data, val_data, test_data)


def read_image(path):
    # imdir = 'path/to/files/'
    ext = ['jpg', 'jpeg', 'png']

    files = []
    images = []
    names = []
    claim = []
    image_id = []
    for e in ext:
        for img in glob.glob(path + "/*." + e):
            files.append(img)

    for f in files:
        names.append(f.split('/')[-1])
        claim.append(int(f.split('/')[-1].split('-')[0]))
        image_id.append(int(f.split('/')[-1].split('-')[2]))
        images.append(cv2.imread(f))

    # images = [cv2.imread(file) for file in files]
    #
    return pd.DataFrame({
        'claim_id': claim,
        'image_files': names,
        'image': images,
        'image_id': image_id
    })


def read_images_corpus(path):
    train = path + '/train/images'
    dev = path + '/val/images'
    test = path + '/test/images'

    train_images = read_image(train)
    dev_images = read_image(dev)
    test_images = read_image(test)

    return (train_images, dev_images, test_images)


def retrieve_data_for_verification(train_text, train_images):
    claim_ids = train_text['claim_id'].values
    claim_ids = list(set(claim_ids))

    claim_data = []
    for claim_id in claim_ids:
        df = train_text.loc[(train_text.claim_id == claim_id)]
        text_evidences = df['Evidence'].values
        image_evidences = train_images.loc[(train_images.claim_id == claim_id)]['image'].values

        claim_object = (df['Claim'].values[0], text_evidences, image_evidences, claim_id)
        claim_data.append(claim_object)

    return claim_data


def get_text_evidences_db(path):
    text_corpus_path = path + '/Corpus3.csv'
    text_evidence = pd.read_csv(text_corpus_path, low_memory=False)
    return text_evidence


def get_text_evidences_sentence_db(path):
    text_corpus_path = path + '/supplementary/Corpus3_sentence_level.csv'
    text_evidence = pd.read_csv(text_corpus_path, low_memory=False)
    return text_evidence


def get_image_evidences_db(path):
    ext = ['jpg', 'jpeg', 'png']

    files = []

    image_evidence = []
    for e in ext:
        for img in glob.glob(path + "/images/*." + e):
            files.append(img)

    for f in tqdm(files):
        name_f = f.split('/')[-1]

        # try:
        #     raw_img = cv2.imread(f)
        #     raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        #     # raw_img = cv2.resize(raw_img, (400, 400), interpolation=cv2.INTER_AREA)
        #     d = (int(name_f.split('-')[0]),
        #          int(name_f.split('-')[1]),
        #          int(name_f.split('-')[2]),
        #          name_f.split('-')[3].split('.')[0],
        #          name_f,
        #          raw_img
        #          )
        #     image_evidence.append(d)
        # except Exception as e:
        #     print(e)
        #     print(name_f)

        try:
            raw_img = cv2.imread(f)
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            # raw_img = cv2.resize(raw_img, (400, 400), interpolation=cv2.INTER_AREA)
            # img = Image.open(f)
            # raw_img = np.asarray(img)
            # img.close()
            
        except Exception as e:
            print(e)
            raw_img = np.zeros((100, 100, 3), np.uint8)
            # raw_img = Image.fromarray(np.zeros((100, 100, 3), np.uint8))
        
        d = (int(name_f.split('-')[0]),
                int(name_f.split('-')[1]),
                int(name_f.split('-')[2]),
                name_f.split('-')[3].split('.')[0],
                name_f,
                raw_img,
                f
            )
        image_evidence.append(d)

    return image_evidence


def get_image_evidences_db_path_only(path):
    ext = ['jpg', 'jpeg', 'png']

    files = []

    image_evidence = []
    for e in ext:
        for img in glob.glob(path + "/images/*." + e):
            files.append(img)

    for f in tqdm(files):
        name_f = f.split('/')[-1]

        try:
            d = (int(name_f.split('-')[0]),
                int(name_f.split('-')[1]),
                int(name_f.split('-')[2]),
                name_f.split('-')[3].split('.')[0],
                name_f,
                f
            )
            image_evidence.append(d)
        except Exception as e:
            # print(e)
            print(name_f)

    return image_evidence


def read_text_retrieval_corpus(path):
    train = path + '/train/text_evidence_qrels_article_level.csv'
    dev = path + '/val/text_evidence_qrels_article_level.csv'
    test = path + '/test/text_evidence_qrels_article_level.csv'

    train_data = pd.read_csv(train, low_memory=False)
    val_data = pd.read_csv(dev, low_memory=False)
    test_data = pd.read_csv(test, low_memory=False)

    return (train_data, val_data, test_data)


def read_sentence_retrieval_corpus(path):
    train = path + '/train/text_evidence_qrels_sentence_level.csv'
    dev = path + '/val/text_evidence_qrels_sentence_level.csv'
    test = path + '/test/text_evidence_qrels_sentence_level.csv'

    train_data = pd.read_csv(train, low_memory=False)
    val_data = pd.read_csv(dev, low_memory=False)
    test_data = pd.read_csv(test, low_memory=False)

    return (train_data, val_data, test_data)


def read_image_retrieval_corpus(path):
    train = path + '/train/img_evidence_qrels.csv'
    dev = path + '/val/img_evidence_qrels.csv'
    test = path + '/test/img_evidence_qrels.csv'

    train_data = pd.read_csv(train, low_memory=False)
    val_data = pd.read_csv(dev, low_memory=False)
    test_data = pd.read_csv(test, low_memory=False)

    return (train_data, val_data, test_data)


def get_origin_dataset(DATA_PATH):
    train_text, dev_text, test_text = read_text_corpus(DATA_PATH)
    train_image, dev_image, test_image = read_images_corpus(DATA_PATH)

    val_claim = retrieve_data_for_verification(dev_text, dev_image)
    train_claim = retrieve_data_for_verification(train_text, train_image)
    test_claim = retrieve_data_for_verification(test_text, test_image)

    return train_claim, val_claim, test_claim


if __name__ == '__main__':
    # train_text, dev_text, test_text = read_text_corpus(DATA_PATH)
    # train_image, dev_image, test_image = read_images_corpus(DATA_PATH)
    #
    # text_evidences = get_text_evidences_db(DATA_PATH)
    # image_evidences = get_image_evidences_db(DATA_PATH)
    #
    # val_claim = retrieve_data_for_verification(dev_text, dev_image)
    # train_claim = retrieve_data_for_verification(train_text, train_image)
    # test_claim = retrieve_data_for_verification(test_text, test_image)

    # sentence_db = get_text_evidences_sentence_db("/home/sonlt/drive/data/mocheg")
    # sentence_db = sentence_db.head(100000)
    # sentence_db.to_csv("Corpus3_sentence_level_tiny.csv", index=False)

    # image_evidences_new = get_image_evidences_db_path_only(DATA_PATH)
    # print(image_evidences_new)

    image_evidences_new = get_image_evidences_db("/home/s2320014/data/mocheg")
    print(len(image_evidences_new))
    hkl.dump(image_evidences_new, '/home/s2320014/data/mocheg/images.hkl', mode='w')
   
    # print("reading image")
    # image_db = hkl.load('/home/s2320014/data/mocheg/images_db.hkl')
    # print(len(image_db))
    pass
