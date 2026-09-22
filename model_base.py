# Generated from: model_base.ipynb
# Converted at: 2026-05-13T11:20:16.073Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import glob
import os
import torch.multiprocessing
# 'spawn' jest bezpieczniejszy niż 'fork' dla PyTorch + pandas/torch.load
torch.multiprocessing.set_start_method('spawn', force=True)
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import tcr_embeddings_path, mhc_embeddings_path, peptide_embeddings_path, all_relations_path, train_path, val_path, test_path, CMV_dataset_path, CNN_tcr_embeddings_path
import pickle
import dhg
import dhg.nn as dnn
from dhg.nn import GCNConv, GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import orthogonal
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import List, Tuple, Optional, Union
import import_ipynb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import average_precision_score
from Evaluator import evaluate
# import Hypergraph as HGipynb


with open(tcr_embeddings_path, "rb") as f:
    tcr_embbedings = pickle.load(f)

with open(mhc_embeddings_path, "rb") as f:
    mhc_embbedings = pickle.load(f)

with open(peptide_embeddings_path, "rb") as f:
    peptide_embbedings = pickle.load(f)

with open(CNN_tcr_embeddings_path, "rb") as f:
    cnn_tcr_embeddings = pickle.load(f)

with open(train_path, "rb") as f:
    Train_data = pickle.load(f)

with open(val_path, "rb") as f:
    Val_data = pickle.load(f)

with open(test_path, "rb") as f: 
    Test_data = pickle.load(f)

with open(CMV_dataset_path, "rb") as f:
    cmv_dataset = pickle.load(f)

map_dict = {"Non-binding":0, "Binding":1}
Train_data["Binding"] = Train_data["Binding"].replace(map_dict)
tcr_embbedings.rename({"Name":"TCR_name"}, axis=1, inplace=True)
cnn_tcr_embeddings.rename({"Name":"TCR_name"}, axis=1, inplace=True)
vals = Train_data["Binding"].value_counts().values
pos_prob = vals[1]/vals[0]
vals

tr = Train_data["Binding"].value_counts().values
v = Val_data["Binding"].value_counts().values
ts = Test_data["Binding"].value_counts().values
print((tr[1]/tr[0]),(v[1]/v[0]),(ts[1]/ts[0]))

tr = Train_data["HLA_MHC"].value_counts().values
v = Val_data["HLA_MHC"].value_counts().values
ts = Test_data["HLA_MHC"].value_counts().values
print((tr[1]/tr[0]),(v[1]/v[0]),(ts[1]/ts[0]))

np.sum(cmv_dataset[["TCR_name", "HLA_MHC"]].groupby("TCR_name").agg("count")["HLA_MHC"].values)/2

cmv_dataset["TCR_name"].drop_duplicates().shape

# ##### Podejście trywialne KNN search


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

df_primary = alpha_beta_pd.copy()
df_primary['MHC'] = 1

# Kopia z etykietą 0 (obiekt powiązany)
df_secondary = alpha_beta_pd.copy()
df_secondary['MHC'] = 0

# Złączenie — każdy Name pojawia się teraz dwa razy
df_final = pd.concat([df_primary, df_secondary], ignore_index=True)

# Posortowanie żeby pary były obok siebie (opcjonalne)
alpha_beta_pd = df_final.sort_values('TCR_name').reset_index(drop=True)

y_all = cmv_dataset[["TCR_name", "Binding"]].sort_values("TCR_name")["Binding"].values
alpha_beta_pd["Y"] = y_all

train_tcr = Train_data["TCR_name"].values
train = alpha_beta_pd[alpha_beta_pd["TCR_name"].isin(train_tcr)]

val_tcr = Val_data["TCR_name"].values
val = alpha_beta_pd[alpha_beta_pd["TCR_name"].isin(val_tcr)]

test_tcr = Test_data["TCR_name"].values
test = alpha_beta_pd[alpha_beta_pd["TCR_name"].isin(test_tcr)]

y_train = train["Y"]
X_train = train.drop(["TCR_name", "Y"], axis=1)
cols = X_train.columns

y_val = val["Y"]
X_val = val.drop(["TCR_name", "Y"], axis=1)

y_test = test["Y"]
X_test = test.drop(["TCR_name", "Y"], axis=1)

k_n = [3,5,7,10,15,20]
scores = []
for i in k_n:
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, y_train)
    score = neigh.score(X_val, y_val)
    scores.append(score)


val.drop("TCR_name", axis=1)

neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(X_train, y_train)
# evaluate_model(model=neigh, ...) — stare API; użyj evaluate() z gotowymi predykcjami

# ##### Podejścei MLP baseline


class model_MLP(nn.Module):
        
    '''Wejście: konkatenacja [ESM2(α), ESM2(β), MHC_embedding] (np. 2560 + 16 = 2576
        wymiarów).'''

    def __init__(self, in_dim, hidden_dim, name = "model_MLP"):
        super().__init__()

        self.projection_layers = nn.Sequential(
            orthogonal(nn.Linear(in_dim, hidden_dim)),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.prediction_layers = nn.Sequential(
            orthogonal(nn.Linear(hidden_dim, (hidden_dim//2))),
            nn.GELU(),
            nn.Dropout(0.3),
            orthogonal(nn.Linear((hidden_dim//2), (hidden_dim//4))),
            nn.GELU(),
            nn.Dropout(0.3),
            orthogonal(nn.Linear((hidden_dim//4), 1)),
        )
    
    def forward(self, X):
        X_ = self.projection_layers(X)
        X_ = self.prediction_layers(X_)
        return X_
        

# ##### Podejście baseline CNN


class TCRCMVConvNet(nn.Module):
    def __init__(self, esm_dim=640, n_filters=16, kernel_sizes=[3,5,7,9],
                 n_mhc=2, mhc_emb_dim=16, dropout=0.3):
        super().__init__()

        self.name = "TCRCMVConvNet"
        
        # Per-chain convolutions
        self.alpha_convs = nn.ModuleList([
            nn.Conv1d(esm_dim, n_filters, k, padding=(k-1)//2)
            for k in kernel_sizes
        ])
        self.beta_convs = nn.ModuleList([
            nn.Conv1d(esm_dim, n_filters, k, padding=(k-1)//2)
            for k in kernel_sizes
        ])
        
        # MHC embedding
        self.mhc_embedding = nn.Embedding(n_mhc, mhc_emb_dim)
        
        # Klasyfikator
        n_chain_features = n_filters * len(kernel_sizes)  # 16 * 4 = 64
        combined_dim = 2 * n_chain_features + mhc_emb_dim  # 64 + 64 + 16 = 144
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        
        # self._init_weights()
    
    def encode_chain(self, x, convs):
        # x: [B, esm_dim, L]
        conv_outs = [F.gelu(conv(x)) for conv in convs]  # [B, n_filters, L]
        concat = torch.cat(conv_outs, dim=1)  # [B, n_filters*K, L]
        pooled = concat.max(dim=2).values  # [B, n_filters*K]
        return pooled
    
    def forward(self, alpha, beta, mhc_id):
        # alpha, beta: [B, esm_dim, L]  (po transpose)
        # mhc_id: [B] (long tensor z indeksami MHC)
        
        alpha_feat = self.encode_chain(alpha, self.alpha_convs)
        beta_feat = self.encode_chain(beta, self.beta_convs)
        mhc_feat = self.mhc_embedding(mhc_id)
        
        combined = torch.cat([alpha_feat, beta_feat, mhc_feat], dim=1)
        logits = self.classifier(combined)  # [B, 1]
        return logits.squeeze(-1)  # [B]
        

# ##### Podejście Grafowe GNN


class TCRCMVGraphNet(nn.Module):
    def __init__(
        self,
        d_esm2=1280,
        d_mhc_onehot=2,
        d_target_flag=4,
        d_hidden=256,
        n_gcn_layers=3,
        dropout_node=0.3,
        dropout_gcn=0.3,
        dropout_classifier=0.3,
    ):
        super().__init__()
        self.name = "TCRCMVGraphNet"
        self.d_hidden = d_hidden
        self.n_gcn_layers = n_gcn_layers
        
        # TCR projection
        self.tcr_projection = nn.Sequential(
            nn.Linear(d_esm2, d_hidden),
            nn.LayerNorm(d_hidden),
        )
        
        # Node fusion (po konkatenacji z MHC i target flag)
        self.node_fusion = nn.Sequential(
            nn.Linear(d_hidden + d_mhc_onehot + d_target_flag, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout_node),
        )
        
        # Stos warstw GCN
        self.gcn_layers = nn.ModuleList([
            GCNConv(d_hidden, d_hidden, use_bn=False, drop_rate=0.0)
            # GATConv(d_hidden, d_hidden, drop_rate=0, use_bn=True)
            # UWAGA: GATConv powoduje crash (segfault) w DHG 0.9.6 na CUDA:
            #   - atten_drop_rate=0.5 domyślnie → NaN gradienty → illegal memory access
            #   - use_bn=True: _init_weights() nie inicjalizuje wewnętrznych warstw DHG
            for _ in range(n_gcn_layers)
        ])
        self.gcn_dropouts = nn.ModuleList([
            nn.Dropout(dropout_gcn)
            for _ in range(n_gcn_layers)
        ])
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout_classifier),
            nn.Linear(d_hidden // 2, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Ostatnia warstwa classifier: gain=1.0 (przed sigmoid w loss)
        nn.init.orthogonal_(self.classifier[-1].weight, gain=1.0)
    
    def forward(self, X_features, graph, target_idx, d_esm2=1280, d_mhc=2):
        # X_features: [N, d_esm2 + d_mhc + d_target_flag] = [N, 1286]
        # graph: dhg.Graph
        # target_idx: scalar (lub batch [B])
        
        # Rozdzielenie cech
        X_esm2 = X_features[:, :d_esm2]
        X_mhc = X_features[:, d_esm2:d_esm2 + d_mhc]
        X_target = X_features[:, d_esm2 + d_mhc:]
        
        # TCR projection
        h_esm2 = self.tcr_projection(X_esm2)  # [N, d_hidden]
        
        # Node fusion
        h_concat = torch.cat([h_esm2, X_mhc, X_target], dim=1)  # [N, d_hidden + 6]
        h_0 = self.node_fusion(h_concat)  # [N, d_hidden]
        
        # GCN stack z residualami
        all_layers = [h_0]
        h = h_0
        for gcn_layer, dropout in zip(self.gcn_layers, self.gcn_dropouts):
            h_new = gcn_layer(h, graph)
            h_new = F.gelu(h_new)
            h_new = dropout(h_new)
            h = h_new + h  # residual
            all_layers.append(h)
        
        # JK aggregation (max-pool po warstwach)
        h_stacked = torch.stack(all_layers, dim=0)  # [n_layers+1, N, d_hidden]
        h_jk = h_stacked.max(dim=0).values  # [N, d_hidden]
        
        # Target node selection
        h_target = h_jk[target_idx]  # [d_hidden] lub [B, d_hidden]
        
        # Classifier
        logits = self.classifier(h_target)  # [1] lub [B, 1]
        return logits.squeeze(-1)  # [] lub [B]

# ##### DataLoader'y


##############################################################
#----------------------!!! DATA LOADERS !!!-------------------
##############################################################
# ===========================================================================
# 1. MLP: cechy tabelaryczne (ESM2 mean-pooled + one-hot MHC)
# ===========================================================================
class MLPDataset(Dataset):
    """Cechy: konkatenacja [ESM2(alpha), ESM2(beta), one-hot MHC]."""
 
    def __init__(self, df, X_TCR, mhc_to_id):
        self.df = df.reset_index(drop=True)
        self.n_mhc = len(mhc_to_id)
        self.mhc_to_id = mhc_to_id
 
        # Pre-load embeddingów do słownika dla szybkiego dostępu
        X_indexed = X_TCR.set_index("TCR_name")
        self.embeddings_dict = {}
        for name in self.df["TCR_name"].unique():
            row = X_indexed.loc[name]
            alpha = np.array(row["Embeddings_alpha"], dtype=np.float32)
            beta = np.array(row["Embeddings_beta"], dtype=np.float32)
            self.embeddings_dict[name] = np.concatenate([alpha, beta])
 
    def __len__(self):
        return len(self.df)
 
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        esm = self.embeddings_dict[row["TCR_name"]]
 
        mhc_onehot = np.zeros(self.n_mhc, dtype=np.float32)
        mhc_onehot[self.mhc_to_id[row["HLA_MHC"]]] = 1.0
 
        features = np.concatenate([esm, mhc_onehot])
        label = float(row["Binding"])
 
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.float32)
 
 
def get_mlp_loader(df, X_TCR, mhc_to_id, batch_size=256, shuffle=False,
                   class_balanced=False, num_workers=4, pin_memory=True):
    """
    Zwraca DataLoader dla MLP.
 
    class_balanced=True dla train (50/50 pos/neg).
    shuffle=False dla val/test.
    """
    dataset = MLPDataset(df, X_TCR, mhc_to_id)
 
    sampler = None
    if class_balanced:
        labels = df["Binding"].values.astype(int)
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
 
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
# ===========================================================================
# 2. CNN: per-residue embeddingi z paddingiem
# ===========================================================================
class CNNDataset(Dataset):
    """
    Per-residue embeddingi dla alpha i beta chain.
    Padding wykonuje collate_cnn (per batch).
    """
 
    def __init__(self, df, X_TCR, mhc_to_id):
        """
        X_TCR : DataFrame z kolumnami:
            TCR_name, Embeddings_alpha, Embeddings_beta.
            Każda komórka *_perres to array [L, d_esm] (różne L per TCR).
        """
        self.df = df.reset_index(drop=True)
        self.mhc_to_id = mhc_to_id
 
        X_indexed = X_TCR.set_index("TCR_name")
        self.embeddings_dict = {}
        for name in self.df["TCR_name"].unique():
            row = X_indexed.loc[name]
            alpha = np.array(row["Embeddings_alpha"], dtype=np.float32)
            beta = np.array(row["Embeddings_beta"], dtype=np.float32)
            self.embeddings_dict[name] = (alpha, beta)
 
    def __len__(self):
        return len(self.df)
 
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        alpha, beta = self.embeddings_dict[row["TCR_name"]]
        mhc_id = self.mhc_to_id[row["HLA_MHC"]]
        label = float(row["Binding"])
 
        return (
            torch.from_numpy(alpha),
            torch.from_numpy(beta),
            torch.tensor(mhc_id, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )
 
 
def collate_cnn(batch):
    """Padding alpha i beta do max długości w batchu. Konwencja Conv1d: [B, C, L]."""
    alphas = [item[0] for item in batch]
    betas = [item[1] for item in batch]
    mhc_ids = torch.stack([item[2] for item in batch])
    labels = torch.stack([item[3] for item in batch])
 
    L_max_a = max(a.shape[0] for a in alphas)
    L_max_b = max(b.shape[0] for b in betas)
    d_esm = alphas[0].shape[1]
 
    alpha_padded = torch.zeros(len(batch), L_max_a, d_esm)
    beta_padded = torch.zeros(len(batch), L_max_b, d_esm)
 
    for i, (a, b) in enumerate(zip(alphas, betas)):
        alpha_padded[i, : a.shape[0]] = a
        beta_padded[i, : b.shape[0]] = b
 
    alpha_padded = alpha_padded.transpose(1, 2)  # [B, d_esm, L_a]
    beta_padded = beta_padded.transpose(1, 2)    # [B, d_esm, L_b]
 
    return alpha_padded, beta_padded, mhc_ids, labels
 
 
def get_cnn_loader(df, X_TCR, mhc_to_id, batch_size=256, shuffle=False,
                   class_balanced=False, num_workers=4, pin_memory=True):
    """Zwraca DataLoader dla CNN."""
    dataset = CNNDataset(df, X_TCR, mhc_to_id)
 
    sampler = None
    if class_balanced:
        labels = df["Binding"].values.astype(int)
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
 
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_cnn,
    )
# ===========================================================================
# 3. GNN: pre-computed podgrafy w plikach .pt
# ===========================================================================
class GNNDataset(Dataset):
    """
    Każdy plik .pt to jeden podgraf z polami:
    X, edge_list, e_weight, target_idx, target_label (i opcjonalnie inne).
    """
 
    def __init__(self, subgraph_dir, pattern="*.pt"):
        """
        subgraph_dir : ścieżka do katalogu z plikami .pt (jeden plik = jeden podgraf).
        pattern : wzorzec glob (domyślnie wszystkie .pt).
        """
        self.files = sorted(glob.glob(os.path.join(subgraph_dir, pattern)))
        if len(self.files) == 0:
            raise ValueError(f"Brak plików .pt w {subgraph_dir}")
 
        # Pre-load etykiet do class_balanced sampling
        # (musimy znać label każdej próbki bez ładowania całego podgrafu)
        self.labels = []
        for f in self.files:
            data = torch.load(f, weights_only=False)
            self.labels.append(int(data["target_relation"]["Binding"]))
        self.labels = np.array(self.labels)
 
    def __len__(self):
        return len(self.files)
 
    def __getitem__(self, idx):
        return torch.load(self.files[idx], weights_only=False)
 
 
# def collate_gnn(batch):
#     """
#     Zwraca listę słowników (bez paddingu).
#     Model w forward pass iteruje po liście grafów.
#     """
#     return {
#         "graphs_data": batch,
#         "labels": torch.stack([
#             torch.tensor(float(b["target_relation"]["Binding"]), dtype=torch.float32)
#             for b in batch
#         ]),
#     }
def collate_gnn(batch):
    x_list, edge_index_list, edge_weight_list, target_idx_list, labels_list = [], [], [], [], []
    num_nodes_offset = 0

    for data in batch:
        # 1. Cechy
        x_list.append(data["X"])

        # 2. Krawędzie — normalizacja do [E, 2] (format DHG: lista par węzłów)
        edge_list = data["edge_list"]
        if not isinstance(edge_list, torch.Tensor):
            edge_list = torch.tensor(edge_list, dtype=torch.long)
        if edge_list.dim() == 2 and edge_list.size(0) == 2:
            edge_list = edge_list.t()   # [2, E] → [E, 2]
        # edge_list jest teraz [E, 2]
        edge_index_list.append(edge_list + num_nodes_offset)

        # 3. Wagi
        e_weight = data["e_weight"]
        if not isinstance(e_weight, torch.Tensor):
            e_weight = torch.tensor(e_weight, dtype=torch.float32)
        edge_weight_list.append(e_weight)

        # 4. target_idx — bezpieczne przez numpy (poprawny pickling)
        V = data["V_TCR_final"]
        czy_target_arr = V["czy_target"].to_numpy()
        target_pos = np.where(czy_target_arr == 1)[0]
        if len(target_pos) == 0:
            raise ValueError("Brak węzła z czy_target==1 w V_TCR_final")
        local_idx = int(target_pos[0])
        target_idx_list.append(local_idx + num_nodes_offset)

        # 5. Label
        labels_list.append(float(data["target_relation"]["Binding"]))

        num_nodes_offset += data["X"].size(0)

    return {
        "x":           torch.cat(x_list, dim=0),
        "edge_index":  torch.cat(edge_index_list, dim=0),  # [total_E, 2]
        "edge_weight": torch.cat(edge_weight_list, dim=0),
        "target_idx":  torch.tensor(target_idx_list, dtype=torch.long),
        "labels":      torch.tensor(labels_list, dtype=torch.float32),
    }
 
 
def get_gnn_loader(subgraph_dir, batch_size=64, shuffle=False,
                   class_balanced=False, num_workers=4, pin_memory=True,
                   pattern="*.pt"):
    dataset = GNNDataset(subgraph_dir, pattern=pattern)
    sampler = None
    if class_balanced:
        labels = dataset.labels
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,  # 🔥 kluczowe
        collate_fn=collate_gnn,
    )


# ===========================================================================
# -------------------!!! Ładowanie Danych !!!--------------------------------
# ===========================================================================
mhc_list = Train_data["HLA_MHC"].value_counts().index.to_numpy()
mhc_id = {f"{i}":j for i, j in zip(mhc_list, range(0, len(mhc_list)))}

# ===========================================================================
# -------------------!!! MLP DataLoader !!!--------------------------------
# ===========================================================================
train_loader_mlp = get_mlp_loader(Train_data, tcr_embbedings, mhc_id, class_balanced=True, num_workers=0)

# ===========================================================================
# -------------------!!! CNN DataLoader !!!--------------------------------
# ===========================================================================
train_loader_cnn = get_cnn_loader(Train_data, cnn_tcr_embeddings, mhc_id, class_balanced=True, num_workers=0)

# ===========================================================================
# -------------------!!! GNN DataLoader !!!--------------------------------
# ===========================================================================
train_loader_gnn = get_gnn_loader(subgraph_dir="Graphs/Training_graphs", num_workers=0)

graph_data = train_loader_gnn.dataset[0]  # pierwszy podgraf
graph_data.keys()
# V_TCR_final = graph_data["V_TCR_final"]
# V_TCR_final.index.to_list()

# ##### Pętla ucząca


# ===========================================================================
# -------------------!!! Pętla ucząca !!!--------------------------------
# ===========================================================================
# def step(model, batch, criterion, device):
#     if isinstance(batch, dict):
#         labels = batch["labels"].to(device)
#         logits_list = []
#         for graph_data in batch["graphs_data"]:
#             # --- Cechy węzłów ---
#             X = graph_data["X"].to(device)
#             num_nodes = X.shape[0]

#             # --- Krawędzie (już lokalne) ---
#             edge_list = graph_data["edge_list"]
#             if not isinstance(edge_list, torch.Tensor):
#                 edge_list = torch.tensor(edge_list, dtype=torch.long)
#             e_weight = graph_data["e_weight"]
#             if not isinstance(e_weight, torch.Tensor):
#                 e_weight = torch.tensor(e_weight, dtype=torch.float32)
#             # edge_list = edge_list.to(device)
#             # e_weight = e_weight.to(device)

#             # --- Budowa grafu DHG ---
#             graph = dhg.Graph(num_nodes, edge_list, e_weight)
#             graph.to(device)
#             # --- Pobranie globalnego target_idx i listy node_ids ---
#             V_TCR_final = V_TCR_final = graph_data["V_TCR_final"]
#             target_val = int(V_TCR_final[V_TCR_final["czy_target"]==1].index.to_list()[0])
#             node_ids = torch.tensor(V_TCR_final.index.to_list())

#             # --- Mapowanie na indeks lokalny (bez żadnych ifów) ---
#             local_idx = (node_ids == target_val).nonzero(as_tuple=True)[0].item()
#             tidx = torch.tensor([local_idx], device=device)

#             # --- Forward ---
#             logit = model(X, graph, tidx)
#             logits_list.append(logit)

#         logits = torch.stack(logits_list).view(-1)  
#     elif len(batch) == 2:
#         features, labels = [t.to(device) for t in batch]
#         logits = model(features).squeeze(-1)
#     else:
#         alpha, beta, mhc_id, labels = [t.to(device) for t in batch]
#         logits = model(alpha, beta, mhc_id)

#     loss = criterion(logits, labels)
#     return loss, logits, labels
def step(model, batch, criterion, device):
    # Disjoint batch — jeden duży graf ze wszystkich podgrafów w batchu
    if isinstance(batch, dict) and "x" in batch and "edge_index" in batch:
        labels     = batch["labels"].to(device)
        x          = batch["x"].to(device)
        target_idx = batch["target_idx"].to(device)

        # edge_index: [total_E, 2] — konwersja do List[List[int]] wymaganej przez DHG
        ei = batch["edge_index"]                 # [total_E, 2]
        e_list = ei.tolist()                     # [[src, dst], ...] — format DHG

        e_weight = batch["edge_weight"]
        if not isinstance(e_weight, torch.Tensor):
            e_weight = torch.tensor(e_weight, dtype=torch.float32)

        graph = dhg.Graph(x.size(0), e_list, e_weight.tolist())
        graph.to(device)

        logits = model(x, graph, target_idx)
        loss   = criterion(logits, labels)
        return loss, logits, labels

    # Fallback: stary format per-graf (lista podgrafów w pętli)
    elif isinstance(batch, dict) and "graphs_data" in batch:
        labels = batch["labels"].to(device)
        logits_list = []
        for graph_data in batch["graphs_data"]:
            X         = graph_data["X"].to(device)
            num_nodes = X.size(0)

            edge_list = graph_data["edge_list"]
            if not isinstance(edge_list, torch.Tensor):
                edge_list = torch.tensor(edge_list, dtype=torch.long)
            if edge_list.dim() == 2 and edge_list.size(0) == 2:
                edge_list = edge_list.t()        # [2, E] → [E, 2]
            e_list = edge_list.tolist()

            e_weight = graph_data["e_weight"]
            if not isinstance(e_weight, torch.Tensor):
                e_weight = torch.tensor(e_weight, dtype=torch.float32)

            graph = dhg.Graph(num_nodes, e_list, e_weight.tolist())
            graph.to(device)

            V_TCR_final = graph_data["V_TCR_final"]
            target_val  = int(V_TCR_final[V_TCR_final["czy_target"] == 1].index.to_list()[0])
            node_ids    = torch.tensor(V_TCR_final.index.to_list())
            local_idx   = (node_ids == target_val).nonzero(as_tuple=True)[0].item()
            tidx        = torch.tensor([local_idx], device=device)

            logits_list.append(model(X, graph, tidx))

        logits = torch.stack(logits_list).view(-1)
        loss   = criterion(logits, labels)
        return loss, logits, labels

    # MLP: batch = (features, labels)
    elif len(batch) == 2:
        features, labels = [t.to(device) for t in batch]
        logits = model(features).squeeze(-1)

    # CNN: batch = (alpha, beta, mhc_id, labels)
    else:
        alpha, beta, mhc_id, labels = [t.to(device) for t in batch]
        logits = model(alpha, beta, mhc_id)

    loss = criterion(logits, labels)
    return loss, logits, labels


def train_model(model, train_loader, val_loader, num_pos, num_neg, device,
                n_epochs=150, patience=15, lr=1e-3, save_path=None):
    """Trening modelu z early stopping na val PR-AUC."""
    model = model.to(device)

    pos_weight = torch.tensor([num_neg / num_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_val_pr_auc = 0.0
    patience_counter = 0
    history = []

    for epoch in range(n_epochs):
        # Trening
        model.train()
        train_loss = 0
        for num, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss, _, _ = step(model, batch, criterion, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Walidacja
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                _, logits, labels = step(model, batch, criterion, device)
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_pr_auc = average_precision_score(all_labels, all_probs)

        scheduler.step(val_pr_auc)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_pr_auc": val_pr_auc})
        print(f"Epoch {epoch:3d} | loss={train_loss:.4f} | val_PR-AUC={val_pr_auc:.4f}")

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return {"best_val_pr_auc": best_val_pr_auc, "history": history}

if __name__ == "__main__":
    val = Train_data["Binding"].value_counts().values
    n_pos_mlp = val[1]
    n_neg_mlp = val[0]
    results_mlp = train_model(
        model=model_MLP(in_dim=1282, hidden_dim=256),
        train_loader=get_mlp_loader(Train_data, X_TCR=tcr_embbedings, mhc_to_id=mhc_id, class_balanced=True, num_workers=0),
        val_loader=get_mlp_loader(Val_data, X_TCR=tcr_embbedings, mhc_to_id=mhc_id, shuffle=False, class_balanced=False, num_workers=0),
        num_pos=n_pos_mlp, num_neg=n_neg_mlp, save_path="best_mlp.pt", device="cuda:0",
    )
    results_cnn = train_model(
        model=TCRCMVConvNet(),
        train_loader=get_cnn_loader(Train_data, cnn_tcr_embeddings, mhc_to_id=mhc_id, class_balanced=True, num_workers=0),
        val_loader=get_cnn_loader(Val_data, cnn_tcr_embeddings, mhc_to_id=mhc_id, shuffle=False, class_balanced=False, num_workers=0),
        num_pos=n_pos_mlp, num_neg=n_neg_mlp, save_path="best_cnn.pt", device="cuda:0",
    )
    # results_gnn = train_model(
    #     model=TCRCMVGraphNet(),
    #     train_loader=get_gnn_loader("Graphs/Training_graphs", class_balanced=True, num_workers=0),
    #     val_loader=get_gnn_loader("Graphs/Validation_graphs", shuffle=False, num_workers=0),
    #     num_pos=num_pos_gnn, num_neg=num_neg_gnn, save_path="best_gnn2.pt", device="cuda:0"
    # )
# model_mlp_test = model_MLP(in_dim=1282, hidden_dim=256)
# model_mlp_test.load_state_dict(torch.load("best_mlp.pt", weights_only=True))
# evaluate_model(model=model_mlp_test) # Zapytać o wejście do torch.nn