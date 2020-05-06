import numpy as np
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import gc
import pandas as pd

path = r"data/IMDB Dataset.csv"
data = pd.read_csv(path)
print(data["review"][0])
mat = np.array(data["review"][0:20000])
labels = np.array(data["sentiment"] == "positive")
labels = np.array(labels.astype(float)[0:20000])

def clean(word):
    '''
    Removes all punctuation from a word
    '''
    alphabet = "ZXCVBNMASDFGHJKLQWERTYUIOP"
    new = ""
    for letter in word:
        if letter.upper() in alphabet:
            new += letter.upper()
    return new
model = KeyedVectors.load_word2vec_format(r"C:\Users\Alexander\Downloads\GoogleNews-vectors-negative300-SLIM.bin.gz", binary=True)

vecs = []
for review in mat:
    review_list = []
    for i in range(50):
        if i > len(review) - 1:
            review_list.append(np.zeros(300))
        else:
            try:
                review_list.append(model[song[i]])
            except:
                review_list.append(np.zeros(300))
    vecs.append(np.array(review_list))
vecs = np.array(vecs)

np.save("data\wordvecs.npy", vecs)
np.save("data\labels.npy", labels)

