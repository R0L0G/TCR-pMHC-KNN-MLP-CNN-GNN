import sys
import os

sys.path.append(os.path.abspath('MGR'))
#Plik zaiwerjący ścieżki do plików

neg_dir_path = "TRAIT/Neg_obs"
pos_dir_path = "TRAIT/Pos_obs"

neg_pkl_path = "TRAIT/negative_interactions.pkl"
pos_pkl_path = "TRAIT/positive_interactions.pkl"

MHC_path = "TRAIT/MHC_pseudo_seqs.txt"

#----------------------!!! Ostateczne dane do tworzenia grafu !!!------------------------

tcr_embeddings_path = "graph_ready_data/tcr_embeddings.pkl"
mhc_embeddings_path = "graph_ready_data/mhc_embeddings.pkl"
peptide_embeddings_path = "graph_ready_data/peptide_embeddings.pkl"
CNN_tcr_embeddings_path = "CNN_embedings_data/tcr_embeddings_CNN.pkl"
CNN_mhc_embeddings_path = "CNN_embedings_data/mhc_embeddings_CNN.pkl"

all_relations_path = "graph_ready_data/all_relations.pkl"
#------------------------!!! Embeddingi CNN CMV problem !!!-------------------------------

CNN_tcr_embeddings_path = "CNN_embedings_data/tcr_embeddings_CNN.pkl"
CNN_mhc_embeddings_path = "CNN_embedings_data/mhc_embeddings_CNN.pkl"

#------------------------!!! Zbiory Uczące !!!-------------------------------

CMV_dataset_path = "training_ready_data/CMV_Dataset.pkl"
train_path = "training_ready_data/Train_data.pkl"
val_path = "training_ready_data/Val_data.pkl"
test_path = "training_ready_data/Test.data.pkl"

#------------------------!!! Gowote struktury do uczenia grafowego !!!-------------------------------

training_graphs_struct_path = "graph_ready_data/training_graphs_struct.pkl"
validation_graphs_struct_path = "graph_ready_data/validation_graphs_struct.pkl"
test_graphs_struct_path = "graph_ready_data/test_graphs_struct.pkl"

