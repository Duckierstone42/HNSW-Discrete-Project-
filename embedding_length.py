import matplotlib.pyplot as plt

import pickle
embeddings=[]
with open('embeddings10000.pickle',"rb") as fp:
    embeddings = pickle.load(fp)
print(len(embeddings[0]))
embeddings_length=[]
for index,embedding in enumerate(embeddings):
    embeddings_length.append(sum(map(lambda x:x**2,embedding)))
    if (index % 1000 == 0):
        print(index)
plt.bar(range(len(embeddings_length)),embeddings_length)
print(embeddings_length)
plt.show()