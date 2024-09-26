import csv
from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset

from data_prepare import clustering
from graphcare_.task_fn import drug_recommendation_fn, drug_recommendation_mimic4_fn, mortality_prediction_mimic3_fn, \
    readmission_prediction_mimic3_fn, length_of_stay_prediction_mimic3_fn, length_of_stay_prediction_mimic4_fn, \
    mortality_prediction_mimic4_fn, readmission_prediction_mimic4_fn
import pickle
import json
from pyhealth.tokenizer import Tokenizer
import numpy as np
from tqdm import tqdm
import torch
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import networkx as nx
from torch_geometric.utils import to_networkx, from_networkx


# %%

def clustering(task, ent_emb, rel_emb, threshold=0.15, load_cluster=False, save_cluster=True):
    if task == "drugrec" or task == "lenofstay":
        path = "data/pj20/exp_data/ccscm_ccsproc"
    else:
        path = "data/pj20/exp_data/ccscm_ccsproc_atc3"

    if load_cluster:
        with open(f'{path}/clusters_th015.json', 'r', encoding='utf-8') as f:
            map_cluster = json.load(f)
        with open(f'{path}/clusters_inv_th015.json', 'r', encoding='utf-8') as f:
            map_cluster_inv = json.load(f)
        with open(f'{path}/clusters_rel_th015.json', 'r', encoding='utf-8') as f:
            map_cluster_rel = json.load(f)
        with open(f'{path}/clusters_inv_rel_th015.json', 'r', encoding='utf-8') as f:
            map_cluster_inv_rel = json.load(f)

    else:
        cluster_alg = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, linkage='average',
                                              affinity='cosine')
        cluster_labels = cluster_alg.fit_predict(ent_emb)
        print(1)
        cluster_labels_rel = cluster_alg.fit_predict(rel_emb)

        def nested_dict():
            return defaultdict(list)

        map_cluster = defaultdict(nested_dict)

        # 处理实体聚类
        for unique_l in tqdm(np.unique(cluster_labels), desc="Processing entity clusters"):
            for cur in range(len(cluster_labels)):
                if cluster_labels[cur] == unique_l:
                    map_cluster[str(unique_l)]['nodes'].append(cur)

        for unique_l in tqdm(map_cluster.keys(), desc="Calculating entity embeddings"):
            nodes = map_cluster[unique_l]['nodes']
            nodes = np.array(nodes)
            embedding_mean = np.mean(ent_emb[nodes], axis=0)
            map_cluster[unique_l]['embedding'].append(embedding_mean.tolist())

        map_cluster_inv = {}
        for cluster_label, item in map_cluster.items():
            for node in item['nodes']:
                map_cluster_inv[str(node)] = cluster_label

        map_cluster_rel = defaultdict(nested_dict)

        # 处理关系聚类
        for unique_l in tqdm(np.unique(cluster_labels_rel), desc="Processing relation clusters"):
            for cur in range(len(cluster_labels_rel)):
                if cluster_labels_rel[cur] == unique_l:
                    map_cluster_rel[str(unique_l)]['relations'].append(cur)

        for unique_l in tqdm(map_cluster_rel.keys(), desc="Calculating relation embeddings"):
            nodes = map_cluster_rel[unique_l]['relations']
            nodes = np.array(nodes)
            embedding_mean = np.mean(rel_emb[nodes], axis=0)
            map_cluster_rel[unique_l]['embedding'].append(embedding_mean.tolist())

        map_cluster_inv_rel = {}
        for cluster_label, item in map_cluster_rel.items():
            for node in item['relations']:
                map_cluster_inv_rel[str(node)] = cluster_label

        if save_cluster:
            with open(f'{path}/clusters_th015.json', 'w', encoding='utf-8') as f:
                json.dump(map_cluster, f, indent=6)
            with open(f'{path}/clusters_inv_th015.json', 'w', encoding='utf-8') as f:
                json.dump(map_cluster_inv, f, indent=6)
            with open(f'{path}/clusters_rel_th015.json', 'w', encoding='utf-8') as f:
                json.dump(map_cluster_rel, f, indent=6)
            with open(f'{path}/clusters_inv_rel_th015.json', 'w', encoding='utf-8') as f:
                json.dump(map_cluster_inv_rel, f, indent=6)

    return map_cluster, map_cluster_inv, map_cluster_rel, map_cluster_inv_rel


# %%
def load_embeddings(task):
    if task == "drugrec" or task == "lenofstay":
        with open('./graphs/cond_proc/CCSCM_CCSPROC/ent2id.json', 'r') as file:
            ent2id = json.load(file)
        with open('./graphs/cond_proc/CCSCM_CCSPROC/rel2id.json', 'r') as file:
            rel2id = json.load(file)
        with open('./graphs/cond_proc/CCSCM_CCSPROC/entity_embedding.pkl', 'rb') as file:
            ent_emb = pickle.load(file)
        with open('./graphs/cond_proc/CCSCM_CCSPROC/relation_embedding.pkl', 'rb') as file:
            rel_emb = pickle.load(file)

    elif task == "mortality" or task == "readmission":
        with open('./graphs/cond_proc_drug/CCSCM_CCSPROC_ATC3/ent2id.json', 'r') as file:
            ent2id = json.load(file)
        with open('./graphs/cond_proc_drug/CCSCM_CCSPROC_ATC3/rel2id.json', 'r') as file:
            rel2id = json.load(file)
        with open('./graphs/cond_proc_drug/CCSCM_CCSPROC_ATC3/entity_embedding.pkl', 'rb') as file:
            ent_emb = pickle.load(file)
        with open('./graphs/cond_proc_drug/CCSCM_CCSPROC_ATC3/relation_embedding.pkl', 'rb') as file:
            rel_emb = pickle.load(file)

    return ent2id, rel2id, ent_emb, rel_emb


# %%
tasks = [
    "drugrec",
    "mortality",
    "readmission",
    "lenofstay"
]

for task in tqdm(tasks):
    print("Loading embeddings...")
    ent2id, rel2id, ent_emb, rel_emb = load_embeddings(task)
    print("Clustering...")
    map_cluster, map_cluster_inv, map_cluster_rel, map_cluster_inv_rel = clustering(task, ent_emb, rel_emb,
                                                                                    threshold=0.15,
                                                                                    load_cluster=False,
                                                                                    save_cluster=True)
