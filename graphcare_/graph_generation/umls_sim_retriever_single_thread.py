import pickle
import numpy as np
from tqdm import tqdm


def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def find_most_similar_embedding(target_emb, umls_ent_emb):
    max_similarity = -1
    max_index = None
    for idx, umls_emb in enumerate(umls_ent_emb):
        similarity = cosine_similarity(target_emb, umls_emb)
        if similarity > max_similarity:
            max_similarity = similarity
            max_index = idx
    if max_similarity > SIMILARITY_THRESHOLD:
        return max_index
    else:
        return None

def single_thread_mapping(id2emb, umls_ent_emb):
    output_dict = {}
    for key in tqdm(id2emb.keys(), desc='Mapping Processing', position=0):
        output_dict[key] = find_most_similar_embedding(id2emb[key], umls_ent_emb)
    return output_dict


if __name__ == "__main__":
    with open("../../KG_mapping/umls/concept_names.txt", 'r', encoding='utf-8') as f:
        umls_ent = f.readlines()

    umls_ids = []
    umls_names = []

    for line in umls_ent:
        umls_id = line.split('\t')[0]
        umls_name = line.split('\t')[1][:-1]
        umls_names.append(umls_name)
        umls_ids.append(umls_id)

    # Define a similarity threshold
    SIMILARITY_THRESHOLD = 0.7

    with open('../../data/pj20/exp_data/atc3_id2emb.pkl', 'rb') as f:
        atc3_id2emb = pickle.load(f)

    with open('../../data/pj20/exp_data/ccscm_id2emb.pkl', 'rb') as f:
        ccscm_id2emb = pickle.load(f)

    with open('../../data/pj20/exp_data/ccsproc_id2emb.pkl', 'rb') as f:
        ccsproc_id2emb = pickle.load(f)

    # 循环实现加载每个umls_ent_emb_i.pkl的分片
    for i in range(547):
        print("epoch:", i+1, "/ 547")
        with open('../../data/pj20/exp_data/split/umls_ent_emb_{idx}.pkl'.format(idx=i), 'rb') as f:
            umls_ent_emb = pickle.load(f)

        ccscm2umls, ccsproc2umls, atc32umls = {}, {}, {}

        ccscm2umls = single_thread_mapping(ccscm_id2emb, umls_ent_emb)
        ccsproc2umls = single_thread_mapping(ccsproc_id2emb, umls_ent_emb)
        atc32umls = single_thread_mapping(atc3_id2emb, umls_ent_emb)

        with open('../../data/pj20/exp_data/ccscm2umls.pkl', 'wb') as f:
            pickle.dump(ccscm2umls, f)

        with open('../../data/pj20/exp_data/ccsproc2umls.pkl', 'wb') as f:
            pickle.dump(ccsproc2umls, f)

        with open('../../data/pj20/exp_data/atc32umls.pkl', 'wb') as f:
            pickle.dump(atc32umls, f)
