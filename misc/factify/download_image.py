import requests

from read_data import read_data
import os

PATH = 'data'

def download_image(data, type='train'):
    img_path = PATH + '/images_set/{}/'.format(type)
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    for _, d in data.iterrows():
        try:
            id = d['Id']
            url_claim = d['claim_image']
            claim_img = requests.get(url_claim).content
            url_doc_claim = d['document_image']
            doc_claim_img = requests.get(url_doc_claim).content

            f1 = open(img_path + '{}.jpg'.format(id), 'wb')
            f2 = open(img_path + '{}_c.jpg'.format(id), 'wb')

            f1.write(claim_img)
            f2.write(doc_claim_img)

            f1.close()
            f2.close()
        except Exception as e:
            print(e)

if __name__ == '__main__':
    train_text, dev_text, test_text = read_data(PATH)
    download_image(train_text, type="test")
