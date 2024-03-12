import re
import numpy as np
import json
import pandas as pd


data = []
with open('/path/qbench_dev_results.jsonl') as f:
    for line in f:
        # Remove leading and trailing whitespace
         # Extract JSON-like substrings using regular expressions
        json_objects = re.findall(r'\{.*?\}', line)
        
        # Load each JSON object and append to the data list
        for obj in json_objects:
            data.append(json.loads(obj))

root = json.load(open('/path/llvisionqa_dev.json', "r"))

choices = "ABCD"
correct = 0
total = 0
for sample in data:
    options = sample['candidates']
    answer = sample['correct_ans']
    response = sample['response']
    index = choices.find(response)
    if options[index].lower() == answer.lower():
        correct += 1
    total += 1
    
print("Accuracy: {}%".format(correct/total*100))