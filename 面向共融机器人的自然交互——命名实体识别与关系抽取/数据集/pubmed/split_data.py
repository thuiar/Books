import json
import sys
import numpy as np
np.random.seed(233)

rel2id = {}
with open("pubmed.json", encoding="utf-8") as f:
    data = json.load(f)

test_idx = np.random.choice(np.arange(len(data)), 500,replace=False)
new_lines_test_test = []
new_lines_test_train = []
for idx,item in enumerate(data):
    if idx in test_idx:
        new_lines_test_test.append(item)
    else:
        new_lines_test_train.append(item)

with open('pubmed_test_test.json','w') as w:
    json.dump(new_lines_test_test,w,indent=4)
with open('pubmed_test_train.json','w') as w:
    json.dump(new_lines_test_train,w,indent=4)