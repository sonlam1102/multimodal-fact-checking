import argparse

from read_data import get_dataset
from train import ClaimVerificationDataset


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="data")
    args = parser.parse_args()
    return args


def make_analysis(data):
    num_claim = []
    num_text_evidence = []
    num_image_evidence = []
    num_len_claim = []

    max_text_evidence = 0
    max_image_evidence = 0

    max_token_text_evidence = 0
    max_token_claim = 0
    total_token_text_evidence = 0
    total_token_claim = 0

    for d in data:
        num_claim.append(d['claim'])
        temp_te = [x for x in d['text_evidence'] if str(x) != 'nan']
        img_set = []
        if d['image_evidence'] is not None:
            img_set.append(d['image_evidence'])
            if d['claim_image'] is not None:
                img_set.append(d['claim_image'])
        elif d['claim_image'] is not None:
            img_set.append(d['claim_image'])

        z = len(set(str(d['claim']).split()))
        total_token_claim = total_token_claim + z
        if z > max_token_claim:
            max_token_claim = z
        num_len_claim.append(z)

        for k in temp_te:
            l = len(set(k.split()))
            total_token_text_evidence = total_token_text_evidence + l
            if l > max_token_text_evidence:
                max_token_text_evidence = l
        num_text_evidence.append(len(temp_te))
        num_image_evidence.append(len(img_set))

        if len(temp_te) > max_text_evidence:
            max_text_evidence = len(temp_te)
        if len(img_set) > max_image_evidence:
            max_image_evidence = len(img_set)

    print("Max token in claim: {}\n".format(max_token_claim))
    print("Avg. token in claim: {}\n".format(total_token_claim / len(num_claim)))

    print("---- statistic by evidences------")
    non_evidence = 0
    print("Number of text evidences: {}\n".format(sum(int(i) for i in num_text_evidence)))
    print("Number of image evidences: {}\n".format(sum(int(i) for i in num_image_evidence)))
    print("Number of claim has no image: {}\n".format(num_image_evidence.count(0)))
    print("Number of claim has no text: {}\n".format(num_text_evidence.count(0)))
    for i in range(0, len(num_claim)):
        if num_image_evidence[i] == 0 and num_text_evidence[i] == 0:
            non_evidence += 1
    print("Number of claim has no evidence: {}\n".format(non_evidence))
    print("Max text evidence: {}\n".format(max_text_evidence))
    print("Max image evidence: {}\n".format(max_image_evidence))
    print("Max token in text evidence: {}\n".format(max_token_text_evidence))
    print(
        "Avg. token of text evidences: {}\n".format(total_token_text_evidence / sum(int(i) for i in num_text_evidence)))
    print("--------END----------")


if __name__ == '__main__':
    args = parser_args()
    train, val, test = get_dataset(args.path)
    train_claim = ClaimVerificationDataset(train)
    dev_claim = ClaimVerificationDataset(val)
    test_claim = ClaimVerificationDataset(test)

    print('----------Train--------------')
    make_analysis(train_claim.to_list())
    print('----------Dev--------------')
    make_analysis(dev_claim.to_list())
    print('----------Test--------------')
    make_analysis(test_claim.to_list())
