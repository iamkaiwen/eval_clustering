import argparse
import json
from math import pow
from itertools import combinations
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('indir', nargs=1, type=str, metavar='indir', help='json file with clusters of embeddings')
    parser.add_argument('outfile', nargs='?', default='result.json', type=str, metavar='output' ,help='output path to write result')
    parser.add_argument('-m', '--manhattan', action='store_true', help='Use manhattan method', dest='manhattan')
    parser.add_argument('-e', '--euclidean', action='store_true', help='Use euclidean method', dest='euclidean')

    return parser.parse_args()

def eval_minkowski(embedding1, embedding2, q):
    ret = map(lambda x, y: abs(x - y) ** q, embedding1, embedding2)
    return pow(sum(ret), 1.0 / q)

# {
#     “result”: [
#         [{“title”:str, “content”: str, “embedding”:array of floats}, { // result2 }, {}],
#         [ cluster 2 ], 
#         [ cluster 3 ], 
#     ]
# }

def eval_similarity(data, q):
    ret = []
    for cluster in data:
        tmp = 0
        for news1, news2 in combinations(cluster, 2):
            tmp += eval_minkowski(news1["embedding"], news2["embedding"], q)
        ret.append(tmp)
    return ret

def eval_dissimilarity(data, q):
    ret = []
    for cluster1, cluster2 in combinations(data, 2):  # 2 for pairs, 3 for triplets, etc
        tmp = 0
        for news1 in cluster1:
            for news2 in cluster2:
                tmp += eval_minkowski(news1["embedding"], news2["embedding"], q)
        ret.append(tmp)
    return ret

def show_plot_box(result, method, cmpclass):
    data = dict(
                zip(
                    result.keys(),
                    [out[method][cmpclass] for out in result.values()]
                )
            )
    df = pd.DataFrame(data)
    df.plot.box(grid='True')
    plt.title(method + " " + cmpclass)
    current_path = os.path.abspath(".")
    plt.savefig(current_path + "\\out\\" + method + "_" + cmpclass + ".png")

if __name__ == '__main__':
    args = parse_args()
    fin, fout = args.indir[0], args.outfile
    
    result = {}

    for dirpath, dirnames, filenames in os.walk(fin):
        for filename in filenames:
            with open(dirpath + '/' + filename, 'r', encoding='utf-8') as f:
                data = json.load(f)['result']
            
            out = {}

            if args.manhattan:
                out["manhattan"] = {"sim" : eval_similarity(data, 1), "dissim" : eval_dissimilarity(data, 1)}
            
            if args.euclidean:
                out["euclidean"] = {"sim" : eval_similarity(data, 2), "dissim" : eval_dissimilarity(data, 2)}

            result[filename] = out

    if args.manhattan:
        show_plot_box(result, "manhattan", "sim")
        show_plot_box(result, "manhattan", "dissim")
    
    if args.euclidean:
        show_plot_box(result, "euclidean", "sim")
        show_plot_box(result, "euclidean", "dissim")

    with open(fout, 'w+', encoding='utf-8') as f:
        f.write(json.dumps(result))
        f.close()