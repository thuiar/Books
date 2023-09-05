import json
import sys
import numpy as np
np.random.seed(233)

def format_data(relation, relid, item:dict):
    token = item['tokens']
    head_strat = item['h'][-1][0][0]
    head_end = item['h'][-1][0][-1]
    tail_start = item['t'][-1][0][0]
    tail_end = item['t'][-1][0][-1]
    one_data = {
        "relation":relation,
        "sentence":token, 
        "head":{"e1_begin":head_strat, "e1_end":head_end}, 
        "tail":{"e2_begin": tail_start, "e2_end": tail_end},
        "relid":relid}
    return one_data

def format_datas():
    p1 = "pubmed.json"
    with open(p1) as f:
        data = json.load(f)
    rels = list(data.keys())
    new_data = []
    rel_idx = 0
    for k,v in data.items():
        for x in v:
            new_data.append(format_data(k, rel_idx, x))
        rel_idx += 1
    print("totle: {}".format(len(new_data)))
    print("rel: {}".format(len(rels)))
    with open("pubmed.json", "w", encoding="utf-8") as f:
        json.dump(new_data, f)
    return 

def check_relation(data):
    rel = set()
    for i, x in enumerate(data):
        r = x["relation"]
        rel.add(r)
    return rel

if __name__ == "__main__":
    format_datas()




