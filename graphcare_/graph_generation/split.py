import pickle
import math

print(int(math.sqrt(297927)))

with open('../../data/pj20/exp_data/umls_ent_emb_.pkl', 'rb') as f:
    a = pickle.load(f)
    print(len(a))

# 297927
last = 0
offset = int(math.sqrt(297927))
i=0
while (1):
    with open('../../data/pj20/exp_data/split/umls_ent_emb_{}.pkl'.format(i), 'wb') as f:
        pickle.dump(a[last:min(last + offset, len(a))], f)
    if last + offset > len(a):
        break
    last += offset
    i+=1
