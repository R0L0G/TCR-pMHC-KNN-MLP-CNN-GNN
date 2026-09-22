
#Plik do przygotowania danych do tworzenia grfów
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import re
from collections import Counter
from config import neg_dir_path, pos_dir_path, neg_pkl_path, pos_pkl_path, MHC_path
from config import tcr_embeddings_path, mhc_embeddings_path, peptide_embeddings_path, all_relations_path
from EMS2_embbedings import get_sequence_embeddings

dir_neg = Path(neg_dir_path)
dir_pos = Path(pos_dir_path)

# ------ PRZESTRZEŃ NA FUNKCJE --------
def format_mhc(value):
    if value[:2] == "NR":
        value = value[3:len(value)-1]
        gene = value[0]  # 'A' or 'B'
        digits = value[1:]
    else:
        gene = value[0]  # 'A' or 'B'
        digits = value[1:]
    return f"HLA-{gene}{digits[:2]}{digits[2:]}"
 

# -------------------------------------

if (Path(neg_pkl_path).exists() and Path(pos_pkl_path).exists()) == False:
    print("--- Uruchamiam łaczenie plików ---")

    neg_first = pd.read_csv(f"{list(dir_neg.iterdir())[0]}", sep="\t")

    for i, file in enumerate(dir_neg.iterdir()):
        if i == 0:
            continue
        else:
            temp = pd.read_csv(f"{file}", sep="\t")
            neg_first = pd.concat([neg_first,temp], axis=0)

    neg_first.to_pickle(neg_pkl_path)

    pos_first = pd.read_csv(f"{list(dir_pos.iterdir())[0]}", sep="\t")

    for i, file in enumerate(dir_pos.iterdir()):
        if i == 0:
            continue
        else:
            temp = pd.read_csv(f"{file}", sep="\t")
            pos_first = pd.concat([pos_first,temp], axis=0)

    pos_first.to_pickle(pos_pkl_path)

else:
    print("--- Ładuje instniejące pliki .pkl ---")

    with open(neg_pkl_path,"rb") as f:
        df_neg = pickle.load(f)

    with open(Path(pos_pkl_path),"rb") as f:
        df_pos = pickle.load(f)

# ------------------------- Wyciąganie danych pod ProteinBERT encoding ------------------------------
All_df = pd.concat([df_neg, df_pos], axis=0)
TCR_df = pd.concat([df_neg, df_pos], axis=0).drop(["Donor", "pMHC", "Binding", "Count"], axis=1)
TCR_df = TCR_df[['CDR3a', 'CDR3b']].drop_duplicates().reset_index(drop=True)

pMHC_df = pd.concat([df_neg, df_pos], axis=0)[["pMHC"]].drop_duplicates().reset_index(drop=True)
MHC_cell = np.array([])
peptide = np.array([])
for i in range(0, len(pMHC_df)):
    temp = pMHC_df["pMHC"][i]
    temp = re.split("_", temp)
    MHC_cell = np.append(MHC_cell, temp[0])
    peptide = np.append(peptide, temp[1])

pMHC_df["MHC"] = MHC_cell
pMHC_df["Peptide"] = peptide
pMHC_df["HLA_MHC"] = pMHC_df["MHC"].apply(format_mhc)

MHC_pseudo_seqs = pd.read_csv(MHC_path, sep=" ", names=["HLA_MHC", "Pseudo_Seq"])

pMHC_df = pd.merge(
    left=pMHC_df,
    right=MHC_pseudo_seqs,
    how="left",
    on="HLA_MHC"
)

MHC_df = pMHC_df[['MHC', 'HLA_MHC','Pseudo_Seq']].drop_duplicates().reset_index(drop=True)
# pMHC_df["Epitop"] = ["_".join(i.split("_")[2:]) for i in pMHC_df['pMHC'].values]
pMHC_df["Epitop"] = pMHC_df['pMHC'].apply(lambda i:"_".join(i.split("_")[2:]))
All_df["Epitop"] = pMHC_df['pMHC'].apply(lambda i:"_".join(i.split("_")[2:]))
Peptide_df = pMHC_df[["Epitop", 'Peptide']]
names_tcr = [f"TCR_{i}" for i in range(1, len(TCR_df)+1)]
TCR_df["Name"] = names_tcr

All_df = pd.merge(
    left=All_df,
    right=TCR_df,
    how="left",
    on=["CDR3a", "CDR3b"]
)
if (Path(tcr_embeddings_path).exists() and Path(mhc_embeddings_path).exists() and Path(peptide_embeddings_path).exists() and Path(all_relations_path).exists()) == False:

    Data_TCR_alpha = [(i, j) for i, j in zip(TCR_df["Name"], TCR_df["CDR3a"])]
    Data_TCR_beta = [(i, j) for i, j in zip(TCR_df["Name"], TCR_df["CDR3b"])]
    Data_MHC = [(i, j) for i, j in zip(MHC_df["HLA_MHC"], MHC_df["Pseudo_Seq"])]
    Data_Peptide = [(i, j) for i, j in zip(Peptide_df["Epitop"], Peptide_df["Peptide"])]

    TCR_alpha_embeddings = get_sequence_embeddings(data=Data_TCR_alpha, batch_size=1024)
    TCR_alpha_embeddings_df = pd.DataFrame(TCR_alpha_embeddings, columns=["Name", "Embeddings_alpha"])

    TCR_beta_embeddings = get_sequence_embeddings(data=Data_TCR_beta, batch_size=1024)
    TCR_beta_embeddings_df = pd.DataFrame(TCR_beta_embeddings, columns=["Name", "Embeddings_beta"])

    TCR_embeddings = pd.merge(
        left=TCR_alpha_embeddings_df,
        right=TCR_beta_embeddings_df,
        how="left",
        on="Name"
    )

    TCR_df = pd.merge(
        left=TCR_df,
        right=TCR_embeddings,
        how="left",
        on="Name"
    )

    relations = All_df[["Name", "TRAV", "TRAJ", "TRBV", "TRBD", "TRBJ"]] #"Name", "TRAV", "TRAJ", "TRBV", "TRBD", "TRBJ"
    relations = relations.drop_duplicates(subset=["Name"])

    TCR_df = pd.merge(
        left=TCR_df,
        right=relations,
        how="left",
        on="Name"
    )

    TCR_df.to_pickle(tcr_embeddings_path)

    MHC_embeddings = get_sequence_embeddings(data=Data_MHC, batch_size=4)
    MHC_embeddings_df = pd.DataFrame(MHC_embeddings, columns=["HLA_MHC", "MHC_embeddings"])

    MHC_df = pd.merge(
        left=MHC_df,
        right=MHC_embeddings_df,
        how="left",
        on="HLA_MHC"
    )
    MHC_df.to_pickle(mhc_embeddings_path)

    Peptide_embeddings = get_sequence_embeddings(data=Data_Peptide, batch_size=64)
    Peptide_embeddings_df = pd.DataFrame(Peptide_embeddings, columns=["Epitop", "Peptide_embeddings"])

    Peptide_df = pd.merge(
        left=Peptide_df,
        right=Peptide_embeddings_df,
        how="left",
        on="Epitop"
    )
    Peptide_df.to_pickle(peptide_embeddings_path)
    # # Datafreme z wszystkimi danymi potrzebnymi do stworzenia grafu
    All_df = All_df.drop_duplicates().reset_index(drop=True)

    All_df["MHC"] = All_df["pMHC"].apply(lambda x: x.split("_")[0])
    All_df["Epitop"] = All_df['pMHC'].apply(lambda i:"_".join(i.split("_")[2:]))
    All_df = pd.merge(left=All_df, right=MHC_df, how="left", on="MHC").drop(["MHC", "Pseudo_Seq", "MHC_embeddings"], axis=1)
    All_df = All_df.rename(columns={"Name":"TCR_name"})
    All_df["Name"] = All_df[["pMHC", "TCR_name"]].apply(lambda row: "_".join(row["pMHC"].split("_") + [row["TCR_name"]]), axis=1)
    
    All_df.to_pickle(all_relations_path)

else:
    print("WSZYSTKIE NIEZBĘDNE PLIKI SĄ GOTOWE")