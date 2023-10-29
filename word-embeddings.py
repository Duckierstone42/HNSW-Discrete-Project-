import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer
import time
model = SentenceTransformer('all-MiniLM-L6-v2')
sentences= ["I really like math","I really like art", "I really hate science", "I really hate math"]
embeddings = model.encode(sentences)
dimension = embeddings.shape[1]
p = hnswlib.Index(space='cosine',dim=dimension)
p.init_index(max_elements=10000,ef_construction=200,M=16) #What is ef_construction and M? Need to read initial paper
#ef stands for link the number of neigbours per node
p.add_items(embeddings)
p.set_ef(50)
new_sentence = "Math is pretty cool"
new_embedding =  model.encode([new_sentence])
# Fetch k neighbours
start_time = time.time()
for i in range(10000):
    labels, distances = p.knn_query(new_embedding,k=2)
end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)
print(sentences[labels[0][0]])