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
from scipy import stats
from sklearn import preprocessing

# https://medium.com/marketingdatascience/%E8%A7%A3%E6%B1%BApython-3-matplotlib%E8%88%87seaborn%E8%A6%96%E8%A6%BA%E5%8C%96%E5%A5%97%E4%BB%B6%E4%B8%AD%E6%96%87%E9%A1%AF%E7%A4%BA%E5%95%8F%E9%A1%8C-f7b3773a889b
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'figure.max_open_warning': 0})

current_path = os.path.abspath(".")

def get_lcm(arr):
    ans = 1
    for x in arr:
        ans = ans * x // gcd(ans, x)
    return ans

def norm_array(arr):
    norm = np.linalg.norm(arr)
    return (arr/norm).tolist()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('indir', nargs=1, type=str, metavar='indir', help='json file with clusters of embeddings')
    parser.add_argument('outfile', nargs='?', default='result.json', type=str, metavar='output' ,help='output path to write result')
    parser.add_argument('-e', '--euclidean', action='store_true', help='Use euclidean method', dest='euclidean')
    parser.add_argument('-t', '--tsne', action='store_true', help='Output tsne pic', dest='tsne')
    parser.add_argument('-s', '--same', action='append', help='Same', dest='same')
    parser.add_argument('-d', '--diff', action='append', help='Difference', dest='diff')
    parser.add_argument('-ex', '--exact', action='store', help='Exact', dest='exact')

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
        if tmp != 0:
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

def show_plot_box(result, diff, method, cmpclass):
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
    print(lcm)
    # print(get_lcm_input)
    for key, value in data.items():
        data[key] = value * (lcm // len(value))

    # get_lcm_input = [len(value) for key, value in data.items()]
    # print(get_lcm_input)
    
    df = pd.DataFrame(data)
    df.plot.box(grid='True')
    plt.title(method + " " + cmpclass)
    plt.xticks(rotation=10)
    plt.tight_layout()
    dirpath = current_path + "\\cmp_out\\"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    diff_concat = '_'.join(diff)
    plt.savefig(dirpath + "\\" + diff_concat + "_" + method + "_" + cmpclass + ".png")
    plt.close()

def show_tsne(dirpath, filename, data):
    tsne = TSNE()
    # tsne = TSNE(n_components=2, init='pca', random_state=0)
    # pca = PCA(n_components=2)
    X = [news["embedding"] for cluster in data for news in cluster]
    label = [cluster_index for cluster_index, cluster in enumerate(data) for index in range(len(cluster))]
    result = tsne.fit_transform(X)
    # result = pca.fit_transform(X)
    result = result[(np.abs(stats.zscore(result)) < 3).all(axis=1)]
    x_min, x_max = result.min(0), result.max(0)
    X_norm = (result - x_min) / (x_max - x_min)
    plt.figure(figsize=(8, 8))
    jet = plt.cm.get_cmap('jet', len(data))
    for i in range(X_norm.shape[0]):
        plt.plot(X_norm[i, 0], X_norm[i, 1], color=jet(label[i] / len(data)), marker='o', markersize=12)

    plt.title('t-SNE embedding of ' + dirpath.split("\\")[-1] + "_" + filename)
    plt.grid()
    dirpath = current_path + "\\t-SNE_out\\" + dirpath.split("\\")[-1]
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    plt.savefig(dirpath + "\\" + filename + ".png")
    # plt.show()

def have_to_cmp(filename, same, diff, exact):
    if exact:
        if not filename.endswith(exact):
            return False

    if same:
        for s in same:
            if s not in filename:
                return False

    if diff:
        for d in diff:
            if d in filename:
                return True
        return False
    return True

if __name__ == '__main__':
    args = parse_args()
    fin, fout = args.indir[0], args.outfile
    
    result = {}

    for dirpath, dirnames, filenames in os.walk(fin):
        for filename in filenames:
            if not filename.endswith(".json") or filename == "groundtruth.json":
                continue
            try:
                if not have_to_cmp(dirpath + '/' + filename, args.same, args.diff, args.exact):
                    continue

                with open(dirpath + '/' + filename, 'r', encoding='utf-8') as f:
                    print("[filename] " + dirpath + '/' + filename)
                    data = json.load(f)['result']
                
                if args.tsne and have_to_cmp(dirpath + '/' + filename, args.same, args.diff, args.exact):
                    # print('gen t-SNE ' + filename.split('.')[0] + '...')
                    show_tsne(dirpath, filename.split('.')[0], data)

                if args.euclidean and have_to_cmp(dirpath + '/' + filename, args.same, args.diff, args.exact):
                    print('Cal ' + filename.split('.')[0] + '...')
                    out = {}
                    
                    if args.euclidean:
                        out["euclidean"] = {"sim" : norm_array(eval_similarity(data, 2)), "dissim" : norm_array(eval_dissimilarity(data, 2))}
                        print(str(len(out["euclidean"]["sim"])) + " " + str(len(out["euclidean"]["dissim"])))

                    print(dirpath)
                    key = dirpath.split('\\')[-1] + "_" + filename
                    result[key] = out
            except Exception as e:
                # print("*** [error] " + dirpath + '/' + filename)
                print(e)
    
    if result and args.euclidean:
        print('Show euclidean plot box ...')
        show_plot_box(result, args.diff, "euclidean", "sim")
        show_plot_box(result, args.diff, "euclidean", "dissim")

    with open(fout, 'w+', encoding='utf-8') as f:
        f.write(json.dumps(result))
        f.close()