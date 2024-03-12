import re
import numpy as np
import json

questions = json.load(open("/path/ai2d_test.json", "r"))
gen_answers = json.load(open("/path/ai2d_results.json", "r"))
def char_to_int(char):
    return ord(char.upper()) - ord('A')

options = "ABCDE"

pattern = re.compile(r'[A-Z]')
results = [(options.find(ans['answer']) == batch_input['metadata']['correctAnswer']) for ans, batch_input in zip(gen_answers, questions)]

print (f"AI2D Acc: {np.mean(results)*100} %")

