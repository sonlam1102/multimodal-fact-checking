import pandas as pd

if __name__ == '__main__':
    # idx2label = {
    #     4: 'Support_Text',
    #     3: 'Support_Multimodal',
    #     2: 'Insufficient_Text',
    #     1: 'Insufficient_Multimodal',
    #     0: 'Refute',
    # }
    # test = pd.read_csv('predict_test_submission.csv')
    # claim_ids = test['claim_id']
    # claim_pred = test['predict']
    # claim_lb = [idx2label[k] for k in claim_pred]
    #
    # submission = pd.DataFrame({
    #     'Id': claim_ids,
    #     'Category': claim_lb
    # })
    #
    # submission.to_csv('answers.csv', index=False)
    test_no_label = pd.read_csv('data/public_folder/test.csv')
    gd = pd.read_csv('goldlabels_test.csv')
    test_no_label['Category'] = gd['Category']

    test_no_label.to_csv('test_gold.csv', index=False)