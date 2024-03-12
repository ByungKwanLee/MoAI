import glob
import json
import jsonlines
import re
import numpy as np

results = "/path/seed_results.json"
seed = "/path/SEED-Bench/SEED-Bench.json"

categories = {1: "Scene Understanding", 2: "Instance Identity", 3: "Instance Attribute", 4: "Instance Location",
              5: "Instance Counting", 6: "Spatial Relation", 7: "Instance Interaction", 8: "Visual Reasoning",
              9: "Text Recognition"}
category_scores = {}

gen_answers = json.load(open(results, "r"))
gt = json.load(open(seed, "r"))
gt = {x['question_id']: x for x in gt['questions']}

score = 0

for k, v in categories.items():
    category_scores[v] = {'total': 0, 'correct': 0}

for data in gen_answers:
    question_id = data['question_id']
    answer = data['answer'][0].upper()
    category = gt[question_id]['question_type_id']

    if gt[question_id]['answer'] == answer:
        score += 1
        category_scores[categories[category]]['correct'] += 1
    category_scores[categories[category]]['total'] += 1

accuracy = score/len(gen_answers)*100
for k, v in category_scores.items():
    acc =v['correct']/v['total']*100
    print(f"{k}:\t{acc}%")
print("============================")
print(f"Total Accuracy: {accuracy}%")
