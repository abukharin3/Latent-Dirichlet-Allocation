import numpy as np
import matplotlib.pyplot as plt

class LDA:
    '''
    Class to implement Latent Dirichlet Allocation via Gibbs Sampling for
    sentiment Analysis
    '''
    def __init__(self, data, labels):
        self.data = data #Bag of words
        self.labels = labels #Binary
        self.word_topics = self.assign_words()



    def assign_words(self):
        return np.random.randint(low = 0, high = 2, size = np.shape(self.data)) * self.data

    def word_prob(self, index):
        prob1 = np.sum(self.word_topics, axis = 0)[index] / np.sum(self.data, axis = 0)[index]
        return prob1

    def train(self):
        for i in range(len(self.data)):
            # Compute P(topic | D)
            prob_1 = np.dot(self.data[i], self.word_topics[i]) / sum(self.data[i])
            prob_0 = 1 - prob_1
            for j in range(len(self.data[i])):
                if self.data[i][j] != 0:
                    # Compute P(Word | T)
                    w1 = self.word_prob(j)
                    w0 = 1 - w1
                    probs = [w0 * prob_0, w1 * prob_1]
                    arg_max = probs.index(max(probs))
                    self.word_topics[i, j] = arg_max

    def score(self):
        topics = []
        for i in range(len(self.data)):
            # Compute P(topic | D)
            prob_1 = np.dot(self.data[i], self.word_topics[i]) / sum(self.data[i])
            prob_0 = 1 - prob_1
            if prob_1 > prob_0:
                topics.append(1)
            else:
                topics.append(0)
        topics = np.array(topics)

        return sum((topics == labels).astype(float)) / len(topics)

train_path = r"data\train.npy"
labels_path = r"data\labels.npy"
train = np.load(train_path, allow_pickle = True)[0:80]
labels = np.load(labels_path, allow_pickle = True)[0:80]
train = (train > 0).astype(float)



lda = LDA(train, labels)
scores = []
for i in range(7):
    print(i)
    print(lda.score())
    lda.train()
    scores.append(lda.score())
plt.plot(scores)
plt.show()
