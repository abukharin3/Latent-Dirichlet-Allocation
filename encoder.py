import numpy as np
import pandas as pd

class encoder:
    '''
    Given a set of tweets, this class can create a vocabulary
    and encode the tweets as a bag of words
    '''
    def __init__(self, data):
        self.data = self.transform(data) # array [tweet, label]
        self.vocab = self.get_vocab()
        self.data = self.n_gram(data)

    def clean(self, word):
        '''
        Removes all punctuation from a word
        '''
        alphabet = "ZXCVBNMASDFGHJKLQWERTYUIOP1234567890"
        new = ""
        for letter in word:
            if letter.upper() in alphabet:
                new += letter.upper()
        return new

    def discretize(self, sentence):
        # Turns a sentence into a list of words
        sentence_list = []
        for word in sentence.split(" "):
            sentence_list.append(self.clean(word))
        return sentence_list

    def n_gram(self, data):
        new = []
        for i in range(len(data)):
            print(i)
            sentence = data[i]
            new.append(np.array(self.encode(sentence)))
        return(np.array(new))


    def encode(self, sentence):
        '''
        Method to encode a song using bag of word model. song must be
        a list of words.
        '''
        if "<unk>" not in self.vocab:
            self.vocab.append("<unk>")
        vec = np.zeros(len(self.vocab))
        for word in sentence:
            if word in self.vocab:
                index = self.vocab.index(word)
                vec[index] += 1
            else:
                index = self.vocab.index("<unk>")
                vec[index] += 1
        return np.array(vec)

    def get_vocab(self):
        all_words = []
        for i in range(len(self.data)):
            all_words += list(self.data[i])
        return list(set(all_words))

    def transform(self, data):
        new = []
        for i in range(len(data)):
            print(i)
            sentence = self.discretize(data[i])
            new.append(np.array(sentence))
        return np.array(new)

    def save(self):
        train = self.data
        np.save("data/train.npy", train)



path = r"data/IMDB Dataset.csv"
data = pd.read_csv(path)
print(data.head())
mat = np.array(data["review"][0:2000])
labels = np.array(data["sentiment"] == "positive")
labels = np.array(labels.astype(float)[0:2000])
#print(sum(labels)) # 501 positive labels

np.save("data/labels.npy", labels)
e = encoder(mat)
e.save()
