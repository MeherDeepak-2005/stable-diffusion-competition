import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

path = "E:/DiffusionDB/train.csv"
df = pd.read_csv(path)

st = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
z = st.encode(df["Prompt"], show_progress_bar=True)
print(z.shape)
np.savez('E:/DiffusionDB/train-embeddings.npz', embeddings=z)
