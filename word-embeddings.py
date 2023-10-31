import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import csv
import matplotlib.pyplot as plt
import pickle
import os
import random

spaceName = "cosine"
wordList= []
count = 300000
def write_list(a_list):
    with open(f'embeddings{count}.pickle','wb') as fp:
        pickle.dump(a_list,fp)
        print("Done writing to pickle file")

def read_list():
    with open(f'embeddings{count}.pickle','rb') as fp:
        return_list = pickle.load(fp)
        return return_list

model = SentenceTransformer('all-MiniLM-L6-v2')
with open('unigram_freq.csv') as file:
        for i in range(count):
            line = file.readline()
            line = line.split(",")[0]
            wordList.append(line)
        print(f"Succesfully loaded {count} words from Unigrams file into a list")


if os.path.isfile(f"embeddings{count}.pickle"):
    #No need to make an embeddings.csv file
    print("Loaded embeddings from saved pickle file")
    embeddings = read_list()
else:
    
    print("Creating new embeddings...")

    start_time = time.time()
    
    #sentences= ["I really like math","I really like art", "I really hate science", "I really hate math"]
    embeddings = model.encode(wordList)
    write_list(embeddings)

    
    end_time = time.time()
    print(f"Converted all words into embeddings in {end_time - start_time}")

    
dimension = embeddings.shape[1]
p = hnswlib.Index(space=spaceName,dim=dimension)

print("Starting to load HNSW graph...")

if os.path.isfile(f"HNSWmodel{count}"):
    print("Loading model from file")
    p.load_index(f"HNSWmodel{count}", max_elements = 400000)


else:
    print("Creating new model...")
    p.init_index(max_elements=400000,ef_construction=200,M=16) #What is ef_construction and M? Need to read initial paper
    start_time = time.time()

#ef stands for link the number of neigbours per node

    p.add_items(embeddings)
    p.set_ef(50)
    end_time = time.time()
    p.save_index(f"HNSWmodel{count}")
    print(f"Succesfully loaded model in {end_time - start_time} seconds")


new_word1 = "king"

new_word2 = "crown"
new_word1 =  model.encode([new_word1])
new_word2 = model.encode([new_word2])
new_embedding = (new_word2 + new_word1)
print(new_word1[0][0])
print(new_word2[0][0])
print(new_embedding[0][0])
# Fetch k neighbours
start_time = time.time()
for i in range(100):
    labels, distances = p.knn_query(new_embedding,k=1)
end_time = time.time()
elapsed_time = end_time - start_time
# The Elapsed time greatly differs, even with the same model, same data, and same query parameter
print(elapsed_time)
print(wordList[labels[0][0]])




#Start from what graphs I want, then come from there.
#First graph, for each distance metric, each data point will be the time it takes to search in a vectoe database starting with 10,000, then adding 10,000 elements to the database.
#For each version of the model, determine the average time it takes to search up 100 randomly selected elements. 
#Graph in millseconds on the y-axis, and number of elements in the HNSW model on the x-axis.