import torch
import numpy as np
from sentence_transformers import util
import pandas as pd

embeddings = np.load('E:/DiffusionDB/train-embeddings-v2.npz')['embeddings']
df_path = 'E:/DiffusionDB/train.csv'
df = pd.read_csv(df_path)
df.set_index("Index", inplace=True)
# df.to_csv('E:/DiffusionDB/backup.csv')
print('Unfiltered embeddings', len(embeddings))

embedding_buffer = len(embeddings) // 8

filtered_embeddings = []
keep_masks = torch.Tensor([])

for x in range(8):
    embedding_bufferList = embeddings[embedding_buffer * x: embedding_buffer * (x + 1)]
    embedding_bufferedTensors = torch.frombuffer(embedding_bufferList,
                                                 dtype=torch.float).view(len(embedding_bufferList), 384)

    cosine_similarities = util.pytorch_cos_sim(embedding_bufferedTensors, embedding_bufferedTensors)
    mask = (cosine_similarities > 0.95).triu(diagonal=1)
    high_similarity_mask = mask.any(dim=1)
    keep_mask = ~high_similarity_mask
    keep_masks = torch.cat([keep_masks, keep_mask])
    filtered_bufferedEmbeddings = embedding_bufferedTensors[keep_mask]
    filtered_embeddings.extend(filtered_bufferedEmbeddings.tolist())

print("DataFrame before:", len(df))

df['Filter'] = keep_masks.tolist()
df_indexes = df[df['Filter'] == 0.0].index
df.drop(df_indexes, inplace=True)

print("DataFrame after:", len(df))
print("Filtered list", len(filtered_embeddings))
np.savez('E:/DiffusionDB/train-embeddings.npz', embeddings=filtered_embeddings)
df.to_csv('E:/DiffusionDB/train.csv')
