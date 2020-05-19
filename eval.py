import argparse
import json
from math import pow
from math import gcd
from itertools import combinations
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# https://medium.com/marketingdatascience/%E8%A7%A3%E6%B1%BApython-3-matplotlib%E8%88%87seaborn%E8%A6%96%E8%A6%BA%E5%8C%96%E5%A5%97%E4%BB%B6%E4%B8%AD%E6%96%87%E9%A1%AF%E7%A4%BA%E5%95%8F%E9%A1%8C-f7b3773a889b
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

current_path = os.path.abspath(".")

def get_lcm(input):
    ans = 1
    for x in input:
        ans = ans * x // gcd(ans, x)
    return ans

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('indir', nargs=1, type=str, metavar='indir', help='json file with clusters of embeddings')
    parser.add_argument('outfile', nargs='?', default='result.json', type=str, metavar='output' ,help='output path to write result')
    parser.add_argument('-m', '--manhattan', action='store_true', help='Use manhattan method', dest='manhattan')
    parser.add_argument('-e', '--euclidean', action='store_true', help='Use euclidean method', dest='euclidean')
    parser.add_argument('-t', '--tsne', action='store_true', help='Output tsne pic', dest='tsne')
    parser.add_argument('-q', '--query', action='append', help='Query', dest='query')

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
        if len(cluster) >= 2:
            tmp = tmp / (len(cluster) * (len(cluster) - 1) / 2)
        ret.append(tmp)
    return ret

def eval_dissimilarity(data, q):
    ret = []
    for cluster1, cluster2 in combinations(data, 2):  # 2 for pairs, 3 for triplets, etc
        tmp = 0
        for news1 in cluster1:
            for news2 in cluster2:
                tmp += eval_minkowski(news1["embedding"], news2["embedding"], q)
        tmp = tmp / (len(cluster1) * (len(cluster2)) / 2)
        ret.append(tmp)
    return ret

def show_plot_box(result, method, cmpclass):
    data = dict(
                zip(
                    [key.split('.')[0] for key in result.keys()],
                    # result.keys(),
                    [out[method][cmpclass] for out in result.values()]
                )
            )

    # extend to same number
    get_lcm_input = [len(value) for key, value in data.items()]
    lcm = get_lcm(get_lcm_input)
    # print(lcm)
    # print(get_lcm_input)
    for key, value in data.items():
        data[key] = value * (lcm // len(value))
    
    df = pd.DataFrame(data)
    df.plot.box(grid='True')
    plt.title(method + " " + cmpclass)
    plt.xticks(rotation=10)
    plt.tight_layout()
    dirpath = current_path + "\\cmp_out\\"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    plt.savefig(dirpath + "\\" + method + "_" + cmpclass + ".png")

def show_tsne(dirpath, filename, data):
    tsne = TSNE()
    # tsne = TSNE(n_components=2, init='pca', random_state=0)
    # pca = PCA(n_components=2)
    X = [news["embedding"] for cluster in data for news in cluster]
    label = [cluster_index for cluster_index, cluster in enumerate(data) for index in range(len(cluster))]
    result = tsne.fit_transform(X)
    # result = pca.fit_transform(X)
    x_min, x_max = result.min(0), result.max(0)
    X_norm = (result - x_min) / (x_max - x_min)
    plt.figure(figsize=(8, 8))
    jet = plt.cm.get_cmap('jet', len(data))
    for i in range(X_norm.shape[0]):
        plt.plot(X_norm[i, 0], X_norm[i, 1], color=jet(label[i] / len(data)), marker='o', markersize=12)

    plt.title('PCA embedding of ' + filename)
    plt.grid()
    dirpath = current_path + "\\t-SNE_out\\" + dirpath.split("\\")[-1]
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    plt.savefig(dirpath + "\\" + filename + ".png")
    # plt.show()

def have_to_cmp(filename, query):
    if not query:
        return True
    
    for q in query:
        if q not in filename:
            return False
    return True


if __name__ == '__main__':
    args = parse_args()
    fin, fout = args.indir[0], args.outfile
    
    result = {}

    for dirpath, dirnames, filenames in os.walk(fin):
        for filename in filenames:
            with open(dirpath + '/' + filename, 'r', encoding='utf-8') as f:
                data = json.load(f)['result']
            
            if args.tsne and have_to_cmp(filename, args.query):
                print('gen t-SNE ' + filename.split('.')[0] + '...')
                show_tsne(dirpath, filename.split('.')[0], data)

            if (args.manhattan or args.euclidean) and have_to_cmp(filename, args.query):
                print('Cal ' + filename.split('.')[0] + '...')
                out = {}

                if args.manhattan:    
                    out["manhattan"] = {"sim" : eval_similarity(data, 1), "dissim" : eval_dissimilarity(data, 1)}
                
                if args.euclidean:
                    out["euclidean"] = {"sim" : eval_similarity(data, 2), "dissim" : eval_dissimilarity(data, 2)}

                result[filename] = out

    if out and args.manhattan:
        print('Show manhattan plot box ...')
        show_plot_box(result, "manhattan", "sim")
        show_plot_box(result, "manhattan", "dissim")
    
    if out and args.euclidean:
        print('Show euclidean plot box ...')
        show_plot_box(result, "euclidean", "sim")
        show_plot_box(result, "euclidean", "dissim")

    with open(fout, 'w+', encoding='utf-8') as f:
        f.write(json.dumps(result))
        f.close()