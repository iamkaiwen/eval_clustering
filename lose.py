import json
import sys

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    print("[data] " + sys.argv[1])
    data = json.load(f)['result']

with open(sys.argv[2], 'r', encoding='utf-8') as f:
    print("[groundtruth] " + sys.argv[2])
    groundtruth = json.load(f)

total = 0
error = 0
for cluster_idx, cluster in enumerate(data):
    for new in cluster:
        total += 1
        if groundtruth[str(new["index"])][0] != cluster_idx:
            error += 1

print("total:\t" + str(total))
print("error:\t" + str(error))
print("lose:\t" + str(1-error/total))