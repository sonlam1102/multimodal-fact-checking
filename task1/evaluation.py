# from torch import tensor
# from torchmetrics.retrieval import RetrievalRecall

def precision_k(ground_truth, predict_new):
    predict = [1 if j > 1 else j for j in predict_new]
    assert len(ground_truth) == len(predict)
    total_k = 0
    for p in predict:
        total_k = total_k + p

    total_relevant_in_k = 0
    for i in range(0, len(ground_truth)):
        if predict[i] == 1 and predict[i] == ground_truth[i]:
            total_relevant_in_k = total_relevant_in_k + 1

    return total_relevant_in_k / total_k if total_k != 0 else 0


def recall_k(ground_truth, predict_new):
    predict = [1 if j > 1 else j for j in predict_new]
    assert len(ground_truth) == len(predict)
    total_relevant = 0
    for p in ground_truth:
        total_relevant = total_relevant + p

    total_relevant_in_k = 0
    for i in range(0, len(ground_truth)):
        if predict[i] == 1 and predict[i] == ground_truth[i]:
            total_relevant_in_k = total_relevant_in_k + 1

    return total_relevant_in_k / total_relevant if total_relevant != 0 else 0

def f1_k(ground_truth, predict):
    assert len(ground_truth) == len(predict)

    prec = precision_k(ground_truth, predict)
    rec = recall_k(ground_truth, predict)

    return (2*prec*rec) / (prec + rec) if (prec + rec) != 0 else 0


def precision_k(ground_truth, predict_new):
    predict = [1 if j > 1 else j for j in predict_new]
    assert len(ground_truth) == len(predict)
    total_k = 0
    for p in predict:
        total_k = total_k + p

    total_relevant_in_k = 0
    for i in range(0, len(ground_truth)):
        if predict[i] == 1 and predict[i] == ground_truth[i]:
            total_relevant_in_k = total_relevant_in_k + 1

    return total_relevant_in_k / total_k if total_k != 0 else 0


def recall_k(ground_truth, predict_new):
    predict = [1 if j > 1 else j for j in predict_new]
    assert len(ground_truth) == len(predict)
    total_relevant = 0
    for p in ground_truth:
        total_relevant = total_relevant + p

    total_relevant_in_k = 0
    for i in range(0, len(ground_truth)):
        if predict[i] == 1 and predict[i] == ground_truth[i]:
            total_relevant_in_k = total_relevant_in_k + 1

    return total_relevant_in_k / total_relevant if total_relevant != 0 else 0

def f1_k(ground_truth, predict):
    assert len(ground_truth) == len(predict)

    prec = precision_k(ground_truth, predict)
    rec = recall_k(ground_truth, predict)

    return (2*prec*rec) / (prec + rec) if (prec + rec) != 0 else 0


def Precision_k(lst_ground_truth, lst_predict):
    total = 0
    assert len(lst_ground_truth) == len(lst_predict)
    for i in range(0, len(lst_ground_truth)):
        total = total + precision_k(lst_ground_truth[i], lst_predict[i])

    return total / len(lst_ground_truth)


def Recall_k(lst_ground_truth, lst_predict):
    total = 0
    assert len(lst_ground_truth) == len(lst_predict)
    for i in range(0, len(lst_ground_truth)):
        total = total + recall_k(lst_ground_truth[i], lst_predict[i])

    return total / len(lst_ground_truth)


def F1_k(lst_ground_truth, lst_predict):
    total = 0
    assert len(lst_ground_truth) == len(lst_predict)
    for i in range(0, len(lst_ground_truth)):
        total = total + f1_k(lst_ground_truth[i], lst_predict[i])

    return total / len(lst_ground_truth)

def average_precision_k(ground_truth, predict):
    gd_new = []
    k_new = []
    assert len(ground_truth) == len(predict)
    top_k = int(max(predict))

    for k in range(1, top_k+1):
        for i in range(0, len(ground_truth)):
            if predict[i] == k:
                k_new.append(predict[i])
                gd_new.append(ground_truth[i])
    
    assert len(gd_new) == len(k_new)
    assert len(gd_new) == top_k

    ap = 0

    relevant = 0
    for i in range(0, len(k_new)):
        if gd_new[i] == 1:
            relevant += 1
        ap = ap + (relevant / k_new[i])*gd_new[i]
    
    return ap / relevant if relevant > 0 else 0

def mean_average_precision(lst_ground_truth, lst_predict):
    assert len(lst_ground_truth) == len(lst_predict)
    total_ap = 0
    for i in range(0, len(lst_ground_truth)):
        total_ap = total_ap + average_precision_k(lst_ground_truth[i], lst_predict[i])

    return total_ap / len(lst_ground_truth)

if __name__ == '__main__':
    # lst_pred = [
    #     [0, 2, 0, 3, 1, 0, 0],
    #     [0, 1, 0, 3, 2, 0, 0],
    #     [0, 2, 0, 3, 1, 0, 0]
    # ]

    # lst_gt = [
    #     [0, 0, 0, 0, 1, 1, 0],
    #     [0, 1, 0, 1, 1, 0, 0],
    #     [0, 0, 0, 0, 0, 1, 1]
    # ]

    # print(F1_k(lst_gt, lst_pred))
    # print(Precision_k(lst_gt, lst_pred))
    # print(Recall_k(lst_gt, lst_pred))

    lst_pred = [
        [1, 2, 3, 4, 5, 6],
        # [1, 0, 2, 3, 0, 4, 0, 5, 0, 0, 0, 0, 0, 0, 0],
        # [1, 0, 2, 3, 0, 4, 0, 5, 0, 0, 0, 0, 0, 0, 0],
        # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
        # [1, 2, 3, 0, 0, 0, 0, 0, 0]
    ]

    lst_gt = [
        [1, 1, 1, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        # [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0],
        # [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        # [0, 0, 0, 1, 1, 1, 0, 0, 0]
    ]

    print(F1_k(lst_gt, lst_pred))
    print(Precision_k(lst_gt, lst_pred))
    print(Recall_k(lst_gt, lst_pred))
    print(mean_average_precision(lst_gt, lst_pred))

    # indexes = tensor([0, 0, 0, 1, 1, 1, 1])
    # preds = tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
    # target = tensor([False, False, True, False, True, False, True])
    # r2 = RetrievalRecall(top_k=2)
    # print(r2(preds, target, indexes=indexes))