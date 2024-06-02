import torch
from langchain_community.embeddings import OllamaEmbeddings
import torch.nn.functional as F
ollama_emb = OllamaEmbeddings(model = "nomic-embed-text")
t1 = "hai"
t2 = "who is this"
score = round(F.cosine_similarity(torch.tensor(ollama_emb.embed_query(t1)),torch.tensor(ollama_emb.embed_query(t2)),dim=0).item(),8)
print(score)
