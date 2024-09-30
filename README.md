# GraphCare
Code for the paper: [GraphCare: Enhancing Healthcare Predictions with Personalized Knowledge Graphs](https://openreview.net/pdf?id=tVTN7Zs0ml) in ICLR'24.


## Requirements:
``` bash
pip install torch==1.12.0
pip install torch-geometric==2.3.0
pip install pyhealth==1.1.2
pip install scikit-learn==1.2.1
pip install openai==0.27.4
```

---

**We follow the flow of methodology section (Section 3) to explain our implementation.**

**由于原版 `README.md` 文件对于运行的东西比较笼统，所以在此做些详细说明，供学习参考**

**书写的步骤按照代码运行的顺序来编写**

## 1. 特定概念知识图谱的生成
**对于这个步骤，论文中是分别给出两种方式：**
- **一种是基于LLM大语言模型给出医疗概念三元组，该复现代码基于*ChatGPT-4o*模型**<br><br>
- **另一种是从现有的知识图谱中抽取**

### 1.1 通过提示进行基于大语言模型的知识图谱抽取
首先配置好`/graphcare_/graph_generation/ChatGPT.py`文件，换上自己的`OpenaiKey`和对应的`url`，默认是官方`url`<br>
然后运行`graph_gen.ipynb`逐个生成三元组
```bash
/graphcare_/graph_generation/graph_gen.ipynb
```
所有生成的三元组会保存在
```bash
/graphs/{condition/CCSCM,procedure/CCSPROC,drug/ATC3}/{code_id}.txt
```
**注意：** 需要先行创建`condition`等文件夹，后面看到新的文件夹都需要先创建，后续不再赘述，

### 1.2 单词向量化操作
首先需要配置好`/graphcare_/graph_generation/get_emb.py`文件，配置方法同1.1的`ChatGPT.py`
```bash
/graphcare_/graph_generation/get_emb.py
```
然后可以进行单词向量检索，可以同时并行以下文件
```bash
/graphcare_/graph_generation/{cond,proc,drug}_emb_ret.py
```
执行完上面的代码之后，就会在先前生成txt的文件夹下生成对应的`json`文件和`pkl`文件
```bash
/graphs/{condition/CCSCM,procedure/CCSPROC,drug/ATC3}/{ent2id,id2ent,id2rel,rel2id}.json
and
/graphs/{condition/CCSCM,procedure/CCSPROC,drug/ATC3}/{entity_embedding,relation_embedding}.pkl
```
然后运行`/graphs/cond_proc`和`/graph/cond_proc_drug`文件夹下的`merge`文件
```bash
/graphs/cond_proc/CCSCM_CCSPROC/merge.ipynb
and
/graphs/cond_proc_drug/CCSCM_CCSPROC_ATC3/merge.ipynb
```
执行完会分别在对应的`merge.ipynb`的文件夹下存有对应的`pkl`和`json`文件

### 1.3 现有的医疗知识图谱抽取
首先先运行`umls_emb_ret.py`文件，对现有的医疗概念进行单词向量化
```bash
/graphcare_/graph_generation/umls_emb_ret.py
```
运行结束生成的`umls_ent_emb_.pkl`文件存于下列文件夹中
```bash
/data/pj20/exp_data/umls_ent_emb_.pkl
```
然后需要运行`data_prepare.py`文件中的`clustering`聚类函数，对于此，我将`clustering`函数抽取出来，对应为`data_prepare_clustering.py`文件
```bash
data_prepare_clustering.py
```
执行完会生成一系列`clusters`的`json`文件在下列文件夹中
```bash
/data/pj20/exp_data/ccscm_ccsproc
and
/data/pj20/exp_data/ccscm_ccsproc_act3
```
再运行`ehr_emb_ret.py`文件，结合病人的相关实际情况，生成对应的`json`文件
```bash
/graphcare_/graph_generation/ehr_emb_ret.py
```
执行完会生成一系列的`json`文件和`pkl`文件在下列文件夹中
```bash
/data/pj20/exp_data/{ccscm_id2emb,ccsproc_id2emb,atc3_id2emb}.pkl
and
/data/pj20/exp_data/ccscm_ccsproc/{ccscm_id2clus,ccsproc_id2clus}.json
and
/data/pj20/exp_data/ccscm_ccsproc_atc3/{ccscm_id2clus,ccsproc_id2clus,atc3_id2clus}.json
```
然后运行`umls_sim_retriever.py`文件

对于此文件，由于要加载出`umls_ent_emb.pkl`文件，整体文件比较大，而源文件又使用过了多线程，所以如果内存没有一定大小很容易爆内存而被`killed`

所以可以采用单线程或者将`umls_ent_emb.pkl`分片来运行
```bash
/graphcare_/graph_generation/umls_sim_retriever.py
or
/graphcare_/graph_generation/split.py and umls_sim_retriever_single_thread.py
```
执行完得到一系列`pkl`文件，存在下列文件夹中
```bash
/data/pj20/exp_data/{ccscm2umls,ccsproc2umls,atc32umls}.pkl
```
最后，运行`umls_sampling.py`文件
```bash
/KG_mapping/umls_sampling.py
```
生成的`pkl`文件存储于以下文件夹中
```bash
/graphs/{ccscm_umls,ccsproc_umls,atc3_umls}/{first_hop_triples,second_hop_triples}.pkl
```

### 1.4 总的数据预处理
首先需要先准备`MIMIC`数据集，需要准备的`csv`数据集如下：
```bash
MIMIC3: /data/physionet.org/files/mimiciii/1.4/{ADMISSIONS,DIAGNOSES_ICD,LABEVENTS,PATIENTS,PRESCRIPTIONS,PROCEDURES_ICD}.csv
MIMIC4: /data/physionet.org/files/mimiciv/2.0/hosp/{admissions,diagnoses_icd,labevents,patients,prescriptions,procedures_icd}.csv
```
然后运行`data_prepare.py`文件
```bash
data_prepare.py
```
执行完毕之后，会形成一系列的`pkl`文件在下列文件夹中：
```bash
/data/pj20/exp_data/ccscm_ccsproc/
and
/data/pj20/exp_data/ccscm_ccsproc_atc3/
```

---

## 1. Concept-specific Knowledge Graph (KG) Generation
### 1.1 LLM-based KG extraction via prompting
The jupyter notebook to prompt KG for EHR medical code:

``` bash
/graphcare_/graph_generation/graph_gen.ipynb
```
We place sample KGs generated by GPT-4 as 
``` bash
/graphs/{condition/CCSCM,procedure/CCSPROC,drug/ATC3}/{code_id}.txt
```

### 1.2 Subgraph sampling from existing KGs
The script for subgraph sampling from UMLS:
``` bash
/KG_mapping/umls_sampling.py
```
We place 2-hop sample KGs randomly subsampled from UMLS as 
``` bash
/graphs/umls_2hop.csv
```

### 1.3 Word Embedding Retrieval for Nodes & Edges
The jupyter notebooks for word embedding retrieval:
``` bash
/graphcare_/graph_generation/{cond,proc,drug}_emb_ret.ipynb
```
Due to the large size of word embedding, we do not include them in the repo. You can use our script to retrieve it and store it in either 
``` bash
/graphs/cond_proc/{entity_embedding.pkl, relation_embedding.pkl}
or
/graphs/cond_proc_drug/{entity_embedding.pkl, relation_embedding.pkl}
```
depending on the features used for the prediction tasks.

### 1.4 Node & Edge Clustering
The function for node & edge clustering:
``` bash
clustering() in data_prepare.py
```
We place some clustering results (only "_inv" as cluster embedding has large size) in 
``` bash
/clustering/
```

## 2. Personalized Knowledge Graph Composition
``` bash
process_sample_dataset() and process_graph() in data_prepare.py
&
get_subgraph() in graphcare.py
```

## 3. Bi-attention Augmented (BAT) Graph Neural Network
The implementation of our proposed BAT model is in
``` bash
/graphcare_/model.py
```

## 4. Training and Prediction
The creation of task-specific datasets (using PyHealth) is in 
``` bash
data_prepare.py
```
The training and prediction details are in
``` bash
graphcare.py
```

## Run Baseline Models
The scripts running baseline models are placed in 
``` bash
ehr_models.py
```

## Cite **GraphCare**
``` bash
@inproceedings{jiang2023graphcare,
  title={GraphCare: Enhancing Healthcare Predictions with Personalized Knowledge Graphs},
  author={Jiang, Pengcheng and Xiao, Cao and Cross, Adam Richard and Sun, Jimeng},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```

Thanks for your interest in our work! 😊

---