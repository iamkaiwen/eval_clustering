import json
import sys

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    data = json.load(f)["result"]

output = []
for cluster_idx, cluster in enumerate(data):
    output.append([])
    for new in cluster:
        output[cluster_idx].append(new["title"])


name = sys.argv[1].split("\\")
print('title_' + name[1] + "_" + name[2])

json.dump({"output" : output}, open('title_' + name[1] + "_" + name[2], "w", encoding="utf-8"), ensure_ascii=False)