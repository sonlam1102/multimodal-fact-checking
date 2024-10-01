import argparse
import pandas as pd
import torch
import datetime

from sklearn.metrics import f1_score

from read_data import get_dataset
from train import ClaimVerificationDataset, train_model, predict


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--size', type=int, default=-1)
    parser.add_argument('--val', default=True, action='store_true')
    parser.add_argument('--path', type=str, default="/home/s2320014/data")
    parser.add_argument('--claim_pt', type=str, default="roberta-base")
    parser.add_argument('--vision_pt', type=str, default="vit")
    parser.add_argument('--long_pt', type=str, default="longformer")
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--use_test', default=False, action='store_true')
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--n_gpu', type=int, default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser_args()
    train, val, test = get_dataset(args.path)
    train_claim = ClaimVerificationDataset(train)
    dev_claim = ClaimVerificationDataset(val)
    test_claim = ClaimVerificationDataset(test)

    if args.size > -1:
        train_claim = train_claim[0:args.size]
        dev_claim = dev_claim[0:args.size*0.1]
        test_claim = test_claim[0:args.size*0.2]

    if args.test:
        model = torch.load(args.model_path)
    else:
        model, loss, name_pt = train_model(train_claim, batch_size=args.batch_size,
                                     epoch=args.epoch, is_val=args.val, val_data=dev_claim, n_gpu=args.n_gpu,
                                     claim_pt=args.claim_pt, vision_pt=args.vision_pt, long_pt=args.long_pt)

        torch.save(model, 'claim_verification_{}.pt'.format(
            str(datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S"))))

    gt, prd, ids = predict(test_claim, model, args.batch_size, n_gpu=args.n_gpu)
    print("Test result micro: {}\n".format(f1_score(gt, prd, average='micro')))
    print("Test result macro: {}\n".format(f1_score(gt, prd, average='macro')))

    output_df = pd.DataFrame({'claim_id': ids, 'predict': prd, 'ground_truth': gt})
    output_df.to_csv('predict_test.csv')

    gtd, prdd, idsd = predict(dev_claim, model, args.batch_size, n_gpu=args.n_gpu)
    print("Dev result micro: {}\n".format(f1_score(gtd, prdd, average='micro')))
    print("Dev result macro: {}\n".format(f1_score(gtd, prdd, average='macro')))

    output_dfd = pd.DataFrame({'claim_id': idsd, 'predict': prdd, 'ground_truth': gtd})
    output_dfd.to_csv('predict_dev.csv')

    if args.use_test:
        _, prd, ids = predict(test_claim, model, args.batch_size, n_gpu=args.n_gpu)
        output_df = pd.DataFrame({'claim_id': ids, 'predict': prd})
        output_df.to_csv('predict_test_submission.csv')
