import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import csv
import matplotlib.pyplot as plt
import pickle
import os
import random
import sys

spaceName = sys.argv[1]
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



print("Creating new model...")
p.init_index(max_elements=400000,ef_construction=200,M=16) #What is ef_construction and M? Need to read initial paper

#ef stands for link the number of neigbours per node

p.set_ef(50)
end_time = time.time()

times=[]
number=[]
previous_index = 0

for i in range(295):
    subWord = embeddings[previous_index:previous_index + 1000]
    p.add_items(subWord)
    previous_index+=1000
    if i % 10 == 0:
        print(f"On HNSW of size {previous_index}")
    randVals = []
    for i in range(100):
        randVal = random.choice(wordList)
        randEmbedding = model.encode([randVal])
        randVals.append(randEmbedding)
    start_time = time.time()
    
    for i in range(100):
        randEmbedding = randVals[i]
        labels, _ = p.knn_query(randEmbedding,k=1)
    end_time = time.time()
    times.append((end_time-start_time) * 1000)
    number.append(previous_index -1)
    



new_word = "ofoal"

new_embedding =  model.encode([new_word])
# Fetch k neighbours
start_time = time.time()
for i in range(100):
    labels, distances = p.knn_query(new_embedding,k=1)
end_time = time.time()
elapsed_time = end_time - start_time
# The Elapsed time greatly differs, even with the same model, same data, and same query parameter
print(elapsed_time)
print(wordList[labels[0][0]])



fig = plt.figure()
x = [1,2,3,4,5]
y = [2,4,8,16,32]
plt.plot(number,times)

plt.xlabel("Number of elements in HNSW")
plt.ylabel("Time in milliseconds")
plt.title(f"Time Required to search up 100 randomly chosen elements \n in HNSW graphs of varying sizes, \n calculated via a {spaceName} distance metric")
with open(f"data{spaceName}.pickle","wb") as fp:
    pickle.dump([number,times],fp)
plt.show()

    #The above code saves the run so I can do the plotting later in a seperate file

#Start from what graphs I want, then come from there.
#First graph, for each distance metric, each data point will be the time it takes to search in a vectoe database starting with 10,000, then adding 10,000 elements to the database.
#For each version of the model, determine the average time it takes to search up 100 randomly selected elements. 
#Graph in millseconds on the y-axis, and number of elements in the HNSW model on the x-axis.