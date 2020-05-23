from itertools import permutations 
import json
import sys
import os

with open(sys.argv[2], 'r', encoding='utf-8') as f:
    print("[groundtruth] " + sys.argv[2])
    groundtruth = json.load(f)

for dirpath, dirnames, filenames in os.walk("."):
        for filename in filenames:
            if sys.argv[1] not in filename:
                continue

            with open(dirpath + '/' + filename, 'r', encoding='utf-8') as f:
                print("[data]\t" + dirpath + '/' + filename)
                data = json.load(f)['result']

            total = 0
            for cluster_idx, cluster in enumerate(data):
                for new in cluster:
                    total += 1
            
            error = total
            perm = permutations([i for i in range(int(sys.argv[3]))])
            for lookup in perm:
                tmp_error = 0
                for cluster_idx, cluster in enumerate(data):
                    for new in cluster:
                        match = False
                        for real in groundtruth[str(new["index"])]:
                            if lookup[real] == cluster_idx:
                                match = True
                        if not match:
                            tmp_error += 1

                if tmp_error < error:
                    error = tmp_error

            print("total:\t" + str(total))
            print("error:\t" + str(error))
            print("lose:\t" + str(1-error/total))