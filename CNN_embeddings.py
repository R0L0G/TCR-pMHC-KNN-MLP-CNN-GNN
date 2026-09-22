# Generated from: CNN_embeddings.ipynb
# Converted at: 2026-05-05T19:14:59.861Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ##### Osoban tabela embeddingów dla modelu CNN à la NetTCR-2.0


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import tcr_embeddings_path, mhc_embeddings_path, peptide_embeddings_path, all_relations_path, train_path, val_path, test_path, CNN_tcr_embeddings_path, CNN_mhc_embeddings_path
import pickle
from EMS2_embbedings import get_sequence_embeddings

with open(tcr_embeddings_path, "rb") as f:
    tcr_embbedings = pickle.load(f)

with open(mhc_embeddings_path, "rb") as f:
    mhc_embbedings = pickle.load(f)
    
with open(train_path, "rb") as f:
    Train_data = pickle.load(f)


mhc_CMV_data = Train_data["HLA_MHC"].value_counts().index.to_numpy()
cnn_mhc = mhc_embbedings[mhc_embbedings["HLA_MHC"].isin(mhc_CMV_data)]
embeding_set = [(i,j) for i,j in zip(cnn_mhc["HLA_MHC"], cnn_mhc["Pseudo_Seq"])]
embeding_set

Data_TCR_alpha = [(i, j) for i, j in zip(tcr_embbedings["Name"], tcr_embbedings["CDR3a"])]
Data_TCR_beta = [(i, j) for i, j in zip(tcr_embbedings["Name"], tcr_embbedings["CDR3b"])]

TCR_alpha_embeddings = get_sequence_embeddings(data=Data_TCR_alpha, batch_size=1024, pre_residue=True)
TCR_alpha_embeddings_df = pd.DataFrame(TCR_alpha_embeddings, columns=["Name", "Embeddings_alpha"])

TCR_beta_embeddings = get_sequence_embeddings(data=Data_TCR_beta, batch_size=1024, pre_residue=True)
TCR_beta_embeddings_df = pd.DataFrame(TCR_beta_embeddings, columns=["Name", "Embeddings_beta"])
TCR_embeddings_CNN = pd.merge(
        left=TCR_alpha_embeddings_df,
        right=TCR_beta_embeddings_df,
        how="left",
        on="Name"
    )

mhc_cnn_embeddings = get_sequence_embeddings(embeding_set, batch_size=1, pre_residue=True)
mhc_cnn_embeddings = get_sequence_embeddings(embeding_set, batch_size=1, pre_residue=True)

MHC_embeddings_df = pd.DataFrame(mhc_cnn_embeddings, columns=["HLA_MHC", "CNN_MHC_embeddings"])
MHC_embeddings_df

# -------------------!!! Zapisywanie nowych embeddingów CNN !!!------------------------------

MHC_embeddings_df.to_pickle(CNN_mhc_embeddings_path)
TCR_embeddings_CNN.to_pickle(CNN_tcr_embeddings_path)