import torch
import numpy as np

model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t30_150M_UR50D")

# plik zawierający funkcję generującą embeddingi
def get_sequence_embeddings(data: list, batch_size:int, pre_residue = False): #seqs powinien wyglądac jak list(zip(nazwa, sekwencja))
    iter = 0
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    all_embeddings = []
    lables = []
    count = 0
    
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i+batch_size]
        _, _, batch_tokens = batch_converter(batch_data)

        seqs = []

        for i in range(0, len(batch_data)):
            seqs.append(batch_data[i][1])
            lables.append(batch_data[i][0])

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[30], return_contacts=False)

        token_embeddings = results["representations"][30]
        
        if pre_residue == False:
            for num, seq in enumerate(seqs):
                seq_len = len(seq)
                seq_embedding = token_embeddings[num, 1:seq_len+1].mean(0).cpu().numpy()
                all_embeddings.append(seq_embedding)
            print(f"results for batch {count}", len(lables))
            count += 1
        else:
            for num, seq in enumerate(seqs):
                seq_len = len(seq)
                seq_embedding = token_embeddings[num, 1:seq_len+1]
                all_embeddings.append(seq_embedding)
            print(f"results for batch {count}", len(lables))
            count += 1

        # if iter > 2:
        #     # print(batch_tokens)
        #     print(seq_embedding.shape, type(seq_embedding))
        #     print(len(all_embeddings), len(all_embeddings[0]))
        #     return [(i,np.array(j)) for i, j in zip(lables, all_embeddings)]
        # iter +=1

    return [(i,np.array(j)) for i, j in zip(lables, all_embeddings)]