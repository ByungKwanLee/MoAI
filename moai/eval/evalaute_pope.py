import os
import json

answer_file = "/path/pope_test_results.json"
annotation_dir = "/path/POPE"
pope_results = {}
pope_results['adversarial'] = None
pope_results['popular'] = None
pope_results['random'] = None

answers = json.load(open(answer_file, "r"))
predictions = {p["question_id"]: p for p in answers}
pred_answers = [p for p in answers]

def eval_pope(answers, label_file):
    label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]
    label_list2 = [json.loads(q) for q in open(label_file, 'r')]
    for answer in answers:
        text = answer['answer'].lower()

        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['answer'] = 'no'
        else:
            answer['answer'] = 'yes'

    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer['answer'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for i, (pred, label) in enumerate(zip(pred_list, label_list)):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    print('%.3f, %.3f, %.3f, %.3f, %.3f' % (f1, acc, precision, recall, yes_ratio) )
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'yes_ratio': yes_ratio}


categories = ['popular', 'random', 'adversarial']

for category in categories:
    cur_answers = [x for x in pred_answers if x['category'] == category]
    cur_answers = sorted(cur_answers, key=lambda x:x["question_id"])
    print('Category: {}, # samples: {}'.format(category, len(cur_answers)))

    if len(cur_answers) == 0:
        continue

    pope_results[category] = eval_pope(cur_answers, os.path.join(annotation_dir, f"coco_pope_{category}.json"))
    print("----------------------------------")