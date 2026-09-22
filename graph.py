# Generated from: graph.ipynb
# Converted at: 2026-05-09T19:34:38.182Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import seaborn as sns
import hnswlib
from config import tcr_embeddings_path, mhc_embeddings_path, peptide_embeddings_path, all_relations_path, CMV_dataset_path, train_path, val_path, test_path
import pickle
import torch
import torch.nn.functional as F
import torch
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
import dhg
from typing import List, Tuple, Optional, Union
from itertools import chain

with open(tcr_embeddings_path, "rb") as f:
    tcr_embbedings = pickle.load(f)

with open(mhc_embeddings_path, "rb") as f:
    mhc_embbedings = pickle.load(f)

with open(peptide_embeddings_path, "rb") as f:
    peptide_embbedings = pickle.load(f)
    
with open(all_relations_path, "rb") as f:
    all_relations = pickle.load(f)

with open(CMV_dataset_path, "rb") as f:
    cmv_dataset = pickle.load(f)

with open(train_path, "rb") as f:
    train_data = pickle.load(f)
    
with open(val_path, "rb") as f:
    val_data = pickle.load(f)

with open(test_path, "rb") as f:
    test_data = pickle.load(f)

map_dict = {"Non-binding":0, "Binding":1}
all_relations["Binding"] = all_relations["Binding"].replace(map_dict)
tcr_embbedings.rename({"Name":"TCR_name"}, axis=1, inplace=True)

t = cmv_dataset[cmv_dataset["Binding"]==1][:100].index.to_numpy()
# all_relations[all_relations["TCR_name"] == "TCR_43"]

alpha_pd = pd.DataFrame(np.stack(tcr_embbedings[["Embeddings_alpha"]].values.flatten()))
alpha_pd["index"] = alpha_pd.index
beta_pd = pd.DataFrame(np.stack(tcr_embbedings[["Embeddings_beta"]].values.flatten()))
beta_pd["index"] = beta_pd.index
alpha_beta_pd=pd.merge(
    left=alpha_pd,
    right=beta_pd,
    how="left",
    on="index"
)
alpha_beta_pd["TCR_name"] = tcr_embbedings["TCR_name"].values
alpha_beta_pd.drop("index", axis=1, inplace=True)
# alpha_beta_pd.loc[[1,2,4,5,8,9], alpha_beta_pd.columns != 'TCR_name']

# alpha_beta_pd # 66957 66956
# tcr_embbedings #66957 66956
alpha_beta_pd

Hypergraf_df = all_relations[["Name", "HLA_MHC", "Epitop", "TCR_name"]]
index = Hypergraf_df.index.values
rng = np.random.default_rng()
r_i = rng.choice(index, size=10)
test_df = Hypergraf_df.iloc[r_i,:]
test_df
for i in test_df.columns[1:-1]:
    print(test_df[i].value_counts())

class _SklearnKnnWrapper:
    """Adapter sklearn -> hnswlib-style API: knn_query(q, k) -> (indices, dist)."""
 
    def __init__(self, model):
        self._model = model
 
    def knn_query(self, query, k):
        distances, indices = self._model.kneighbors(query, n_neighbors=k)
        return indices, distances
 
 
def graph_knn_cl(data: pd.DataFrame, n_jobs=-1):
    """
    Deterministyczny zamiennik hnswlib-owego graph_knn.
 
    n_jobs : rdzenie dla sklearn.kneighbors. Ustaw 1 gdy używasz outer
             ThreadPoolExecutor (uniknięcie zagnieżdżania).
 
    Zwraca: (knn_graph, embeddings_norm) — sygnatura jak oryginał.
    """
    embeddings = data.iloc[:, :-1].values  # bez kolumny TCR_name
    n, dim = embeddings.shape
 
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / norms
 
    nn = NearestNeighbors(
        n_neighbors=200,
        metric='cosine',
        algorithm='brute',
        n_jobs=n_jobs,
    )
    nn.fit(embeddings_norm)
 
    return _SklearnKnnWrapper(nn), embeddings_norm
 
 
# =============================================================================
# GŁÓWNA FUNKCJA
# =============================================================================
def graph_edge_chooser_cl(
    data_all,
    target_edges,
    knn_graph,
    norm_embeddings,
    knn_data,
    X_TCR,
    X_MHC,
    output_dir=None,
    k_sim=10,
    n_jobs=8,
    overwrite=False,
    quiet=False,
):
    """
    output_dir : 
        - None (lub "" lub False): TRYB IN-MEMORY. Funkcja zwraca listę dictów
          identyczną z oryginalną wersją {V_TCR_final, H, edge_list, e_weight,
          target_flag, X, target_relation}. Do szybkiego testowania.
        - "ścieżka/do/folderu": TRYB DISK. Każdy edge zapisywany do
          edge_{idx}.pt; funkcja zwraca listę ścieżek.
 
    Pozostałe argumenty bez zmian.
    """
    save_to_disk = bool(output_dir)
 
    if save_to_disk:
        os.makedirs(output_dir, exist_ok=True)
 
    # =========================================================================
    # PRECOMPUTE
    # =========================================================================
    all_mhc = sorted(data_all["HLA_MHC"].unique())
    mhc_to_id = {m: i for i, m in enumerate(all_mhc)}
    n_mhc = len(all_mhc)
    X_TCR_indexed = X_TCR.set_index("TCR_name")
 
    tcr_to_emb_idx = dict(
        zip(knn_data["TCR_name"].values, knn_data.index.values)
    )
    known_tcrs_knn = set(tcr_to_emb_idx.keys())
 
    alpha_mat = np.stack(X_TCR_indexed["Embeddings_alpha"].values)
    beta_mat = np.stack(X_TCR_indexed["Embeddings_beta"].values)
    esm_mat = np.concatenate([alpha_mat, beta_mat], axis=1)
    xtcr_name_to_row = {n: i for i, n in enumerate(X_TCR_indexed.index)}
    known_tcrs_xtcr = set(xtcr_name_to_row.keys())
 
    print_lock = Lock()
 
    def log(msg):
        if not quiet:
            with print_lock:
                print(msg)
 
    # =========================================================================
    # JEDEN EDGE
    # Zwraca: (result, skip_info)
    #   - w trybie disk: result = ścieżka albo None
    #   - w trybie memory: result = dict albo None
    # =========================================================================
    def process_edge(edge):
        if save_to_disk:
            out_path = os.path.join(output_dir, f"edge_{edge}.pt")
            if not overwrite and os.path.exists(out_path):
                return out_path, None
 
        # 1. TARGET
        target_relation = data_all.iloc[edge, :]
        target_tcr_name = target_relation["TCR_name"]
        target_hla = target_relation["HLA_MHC"]
 
        # 2. EMBEDDING TARGETU
        if target_tcr_name not in tcr_to_emb_idx:
            log(f"POMINIĘTO {edge}: brak embeddingu dla {target_tcr_name}")
            return None, (edge, "no_embedding")
 
        target_emb_idx = int(tcr_to_emb_idx[target_tcr_name])
        target_tcr_embedding = norm_embeddings[target_emb_idx:target_emb_idx + 1]
 
        # 3. KNN — pełny 1280-D query (różnica vs oryginał: bez [:, :-1])
        neighbours, cos_dist = knn_graph.knn_query(target_tcr_embedding, k=200)
        cos_dist = np.round(cos_dist[0].astype(np.float32), 4)
        neighbours_names = X_TCR.iloc[neighbours.flatten(), :]["TCR_name"].values
 
        cos_df = (
            pd.DataFrame({"TCR_name": neighbours_names, "cos_sim": cos_dist})
            .drop_duplicates(subset="TCR_name", keep="first")
        )
 
        neigbours_relations = (
            data_all.loc[
                data_all["TCR_name"].isin(neighbours_names),
                ["TCR_name", "HLA_MHC", "Name", "Binding"]
            ]
            .merge(cos_df, on="TCR_name", how="left")
        )
 
        neighbours_tier1 = neigbours_relations[
            neigbours_relations["HLA_MHC"] == target_hla
        ].copy()
 
        # 4. WYMUSZENIE TARGETU
        mask_dup_target = (
            (neighbours_tier1["TCR_name"] == target_tcr_name)
            & (neighbours_tier1["HLA_MHC"] == target_hla)
        )
        neighbours_tier1 = neighbours_tier1.loc[~mask_dup_target].copy()
        neighbours_tier1["czy_target"] = 0
 
        target_to_add = pd.DataFrame([{
            "TCR_name":   target_tcr_name,
            "HLA_MHC":    target_hla,
            "Name":       target_relation["Name"],
            "Binding":    target_relation["Binding"],
            "cos_sim":    0.0,
            "czy_target": 1,
        }])[neighbours_tier1.columns]
 
        V_TCR_final = pd.concat(
            [neighbours_tier1, target_to_add], axis=0, ignore_index=True
        )
 
        n_targets = int((V_TCR_final["czy_target"] == 1).sum())
        if n_targets != 1:
            log(f"POMINIĘTO {edge}: target_count={n_targets}")
            return None, (edge, f"target_count={n_targets}")
 
        # 5. SKIP CHECKS
        names_in_subgraph = V_TCR_final["TCR_name"].tolist()
 
        missing_in_knn = [n for n in names_in_subgraph if n not in known_tcrs_knn]
        if missing_in_knn:
            log(f"POMINIĘTO {edge}: {len(missing_in_knn)} bez embeddingu")
            return None, (edge, f"missing_in_knn={len(missing_in_knn)}")
 
        missing_in_xtcr = [n for n in names_in_subgraph if n not in known_tcrs_xtcr]
        if missing_in_xtcr:
            log(f"POMINIĘTO {edge}: {len(missing_in_xtcr)} bez X_TCR")
            return None, (edge, f"missing_in_xtcr={len(missing_in_xtcr)}")
 
        if target_hla not in mhc_to_id:
            log(f"POMINIĘTO {edge}: HLA_MHC={target_hla} unknown")
            return None, (edge, "unknown_mhc")
 
        # =====================================================================
        # BUDOWA STRUKTURY GRAFU
        # =====================================================================
        N = len(V_TCR_final)
        target_idx = int(V_TCR_final.index[V_TCR_final["czy_target"] == 1][0])
 
        local_emb_idx = [tcr_to_emb_idx[name] for name in V_TCR_final["TCR_name"]]
        local_embeddings = torch.tensor(
            norm_embeddings[local_emb_idx, :-1], dtype=torch.float32
        )
 
        sim = local_embeddings @ local_embeddings.T
        sim.fill_diagonal_(-float("inf"))
        topk_vals, topk_idx = sim.topk(k=min(k_sim, N - 1), dim=1)
 
        edges = {}
        for i in range(N):
            for k in range(topk_idx.shape[1]):
                j = int(topk_idx[i, k])
                key = (min(i, j), max(i, j))
                w = float(topk_vals[i, k])
                edges[key] = max(edges.get(key, w), w)
 
        edge_list = list(edges.keys())
        M = len(edge_list)
 
        if M > 0:
            edge_arr = torch.tensor(edge_list, dtype=torch.long)
            col_idx = torch.arange(M)
            H = torch.zeros(N, M)
            H[edge_arr[:, 0], col_idx] = 1
            H[edge_arr[:, 1], col_idx] = 1
        else:
            H = torch.zeros(N, 0)
 
        e_weight = torch.tensor([edges[e] for e in edge_list], dtype=torch.float32)
 
        A = (H @ H.T > 0).float()
        A.fill_diagonal_(0)
        one_hop = A[target_idx].bool()
        two_hop_raw = (A @ A)[target_idx].bool()
        two_hop = two_hop_raw & ~one_hop
        two_hop[target_idx] = False
 
        target_flag = torch.zeros(N, 4)
        target_flag[target_idx, 0] = 1
        target_flag[one_hop, 1] = 1
        target_flag[two_hop, 2] = 1
        target_flag[(target_flag.sum(1) == 0), 3] = 1
 
        rows = [xtcr_name_to_row[n] for n in V_TCR_final["TCR_name"]]
        X_esm = torch.tensor(esm_mat[rows], dtype=torch.float32)
 
        mhc_id = mhc_to_id[target_hla]
        X_mhc = torch.zeros(N, n_mhc)
        X_mhc[:, mhc_id] = 1.0
 
        X = torch.cat([X_esm, X_mhc, target_flag], dim=1)
 
        # =====================================================================
        # WYNIK — zależny od trybu
        # =====================================================================
        if save_to_disk:
            sample = {
                "V_TCR_final":     V_TCR_final,
                "H":               H,
                "edge_list":       edge_list,
                "e_weight":        e_weight,
                "target_flag":     target_flag,
                "X":               X,
                "target_relation": target_relation,
                "edge_idx":        edge,
            }
            tmp_path = out_path + ".tmp"
            torch.save(sample, tmp_path)
            os.replace(tmp_path, out_path)
            return out_path, None
        else:
            # Format 1:1 z oryginalną funkcją (bez edge_idx)
            result = {
                "V_TCR_final":     V_TCR_final,
                "H":               H,
                "edge_list":       edge_list,
                "e_weight":        e_weight,
                "target_flag":     target_flag,
                "X":               X,
                "target_relation": target_relation,
            }
            return result, None
 
    # =========================================================================
    # WYKONANIE
    # =========================================================================
    results = []
    skipped = []
 
    if n_jobs == 1:
        for edge in target_edges:
            res, skip = process_edge(edge)
            if res is not None:
                results.append(res)
            if skip is not None:
                skipped.append(skip)
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            for res, skip in ex.map(process_edge, target_edges):
                if res is not None:
                    results.append(res)
                if skip is not None:
                    skipped.append(skip)
 
    # Manifest — tylko w trybie disk
    if save_to_disk:
        manifest_path = os.path.join(output_dir, "manifest.txt")
        with open(manifest_path, "w") as f:
            for p in sorted(results):
                f.write(p + "\n")
        print(f"\nZapisano {len(results)} sub-grafów do {output_dir}")
    else:
        print(f"\nZbudowano {len(results)} sub-grafów w pamięci")
 
    if skipped:
        print(f"Pominięto {len(skipped)} edge'y:")
        for e, reason in skipped[:20]:
            print(f"  edge={e}: {reason}")
        if len(skipped) > 20:
            print(f"  ... i {len(skipped) - 20} więcej")
 
    return results
 
 
# =============================================================================
# DATASET DLA PYTORCH DATALOADER
# =============================================================================
class TCRSubgraphDataset(Dataset):
    """
    Dataset zwracający sub-grafy zapisane przez graph_edge_chooser (tryb disk).
 
    Konstruktor przyjmuje:
      - ścieżkę do katalogu (znajdzie manifest.txt lub przeskanuje edge_*.pt)
      - ścieżkę do manifest.txt
      - listę ścieżek
 
    UWAGA: subgraphy mają RÓŻNE N i M — użyj `collate_fn=tcr_subgraph_collate`.
    """
 
    def __init__(self, source):
        if isinstance(source, (list, tuple)):
            self.paths = list(source)
        elif os.path.isdir(source):
            manifest = os.path.join(source, "manifest.txt")
            if os.path.exists(manifest):
                with open(manifest) as f:
                    self.paths = [line.strip() for line in f if line.strip()]
            else:
                self.paths = sorted(
                    os.path.join(source, f)
                    for f in os.listdir(source)
                    if f.startswith("edge_") and f.endswith(".pt")
                )
        elif os.path.isfile(source):
            with open(source) as f:
                self.paths = [line.strip() for line in f if line.strip()]
        else:
            raise ValueError(f"Nie potrafię zinterpretować source: {source}")
 
        if len(self.paths) == 0:
            raise RuntimeError(f"Pusty Dataset — brak plików w {source}")
 
    def __len__(self):
        return len(self.paths)
 
    def __getitem__(self, idx):
        return torch.load(self.paths[idx], weights_only=False)
 
 
def tcr_subgraph_collate(batch):
    """Trywialny collate: zwraca listę próbek (sub-grafy mają różne wymiary)."""
    return batch

knn_graph, norm_embeddings = graph_knn_cl(alpha_beta_pd, n_jobs=1)

# r_1 = graph_edge_chooser_cl(
#     cmv_dataset,
#     target_edges=[t[0]],
#     knn_graph=knn_graph,
#     norm_embeddings=norm_embeddings,
#     knn_data=alpha_beta_pd,
#     X_TCR=tcr_embbedings,
#     X_MHC=mhc_embbedings,
#     # output_dir nie podany -> in-memory
#     n_jobs=1,
# )   #knn_graph,norm_embeddings,

# r_2 = graph_edge_chooser_cl(
#     cmv_dataset,
#     target_edges=[t[0]],
#     knn_graph=knn_graph,
#     norm_embeddings=norm_embeddings,
#     knn_data=alpha_beta_pd,
#     X_TCR=tcr_embbedings,
#     X_MHC=mhc_embbedings,
#     output_dir="test",
#     n_jobs=1,
# )   #knn_graph,norm_embeddings,

train_positive = train_data[train_data["Binding"]==1]
train_negative = train_data[train_data["Binding"]==0]
train_positive_count = train_positive.shape[0]
train_negative = train_negative.sample(n=2*train_positive_count, replace=False, random_state=42)
training_data = pd.concat([train_positive, train_negative], axis=0)
training_indexs = cmv_dataset[cmv_dataset["Name"].isin(training_data["Name"].values)].index.to_numpy()

# training_graphs_struct = graph_edge_chooser_cl(cmv_dataset, target_edges=training_indexs, knn_data=alpha_beta_pd, X_TCR = tcr_embbedings, X_MHC = mhc_embbedings
#                                                , knn_graph=knn_graph, norm_embeddings=norm_embeddings, output_dir="Graphs/Training_graphs", n_jobs=16)
# print("training_graphs_struct done")

val_positive = val_data[val_data["Binding"]==1]
val_negative = val_data[val_data["Binding"]==0]
val_positive_count = val_positive.shape[0]
val_positive_count
val_negative = val_negative.sample(n=2*val_positive_count, replace=False, random_state=42)
validation_data = pd.concat([val_positive, val_negative], axis=0)
validation_indexs = cmv_dataset[cmv_dataset["Name"].isin(validation_data["Name"].values)].index.to_numpy()
validation_indexs

# validation_graphs_struct = graph_edge_chooser_cl(cmv_dataset, target_edges=validation_indexs, knn_data=alpha_beta_pd, X_TCR = tcr_embbedings, X_MHC = mhc_embbedings
#                                                , knn_graph=knn_graph, norm_embeddings=norm_embeddings, output_dir="Graphs/Validation_graphs", n_jobs=8)
# print("validation_graphs_struct done")

test_positive = test_data[test_data["Binding"]==1]
test_negative = test_data[test_data["Binding"]==0]
test_positive_count = test_positive.shape[0]
test_positive_count
test_negative = test_negative.sample(n=2*test_positive_count, replace=False, random_state=42)
testing_data = pd.concat([test_positive, test_negative], axis=0)
test_indexs = cmv_dataset[cmv_dataset["Name"].isin(testing_data["Name"].values)].index.to_numpy()
test_indexs

# test_graphs_struct = graph_edge_chooser_cl(cmv_dataset, target_edges=test_indexs, knn_data=alpha_beta_pd, X_TCR = tcr_embbedings, X_MHC = mhc_embbedings
#                                                , knn_graph=knn_graph, norm_embeddings=norm_embeddings, output_dir="Graphs/Test_graphs", n_jobs=8)
# print("test_graphs_struct done")

# training_graphs_struct.to_pickle(training_graphs_struct_path)
# validation_graphs_struct.to_pickle(validation_graphs_struct_path)
# test_graphs_struct.to_pickle(test_graphs_struct_path)

train_idx_name = cmv_dataset.iloc[training_indexs,:]["Name"].values
val_idx_name = cmv_dataset.iloc[validation_indexs,:]["Name"].values
test_idx_name = cmv_dataset.iloc[test_indexs,:]["Name"].values

with open('indexy_graphs/train_index.pkl','wb') as f:
    pickle.dump(train_idx_name, f)

with open('indexy_graphs/val_index.pkl','wb') as f:
    pickle.dump(val_idx_name, f)

with open('indexy_graphs/test_index.pkl','wb') as f:
    pickle.dump(test_idx_name, f)

print("Wszystko poprawnie zapisane!!!!!!!")