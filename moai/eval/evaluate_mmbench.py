import re
import numpy as np
import json
import pandas as pd

df = pd.read_excel("/path/mmbench_test.xlsx")
root = pd.read_table('/path/MMBench_TEST_EN_legacy.tsv')
# correct_predictions = (df['prediction'] == df['answer']).sum()
# total_predictions = len(df)
# accuracy = correct_predictions / total_predictions * 100

# print("Accuracy: {:.2f}%".format(accuracy))

root = root.drop(columns=['hint', 'category', 'source', 'image', 'comment', 'l2-category'])
root.insert(6, 'prediction', None)

root.loc[root['index'] == df['index'], 'prediction'] = df['prediction']

root.to_excel("/path/mmbench_testing.xlsx", index=False, engine='openpyxl')
print(root)