import numpy as np

class NaiveBayes:
    '''
    Naive Bayes classifier for tweets. Given a set of tweets, learn
    to classify if a tweet is positive or negative.
    '''
    def __init__(self, data, labels):
        self.data = data # [num tweets, [vec, scale]]
        self.labels = labels # 0 or 1
        self.vocab_len = len(self.data[0])
        self.counts = [len(data) - sum(labels), sum(labels)]
        self.word_counts = self.get_word_counts(self.data)
        self.probs = self.learn_params(self.data)

    def get_word_counts(self, data):
        word_counts = np.zeros(self.vocab_len)
        for tweet in data:
            word_counts += tweet[0]
        return word_counts + np.ones(self.vocab_len)


    def learn_params(self, data):
        '''
        Method to learn the posterior probabilities, probability of a word given
        a genre and the prior probabilities
        '''

        word_counts = self.word_counts
        probs = np.ones([len(self.counts), self.vocab_len]) / self.vocab_len
        for i in range(len(data)):
            rating = int(self.labels[i])
            probs[rating] += data[i]  / (word_counts[rating] +  self.vocab_len)
        return probs

    def predict(self, x):
        x = np.array(x)
        category = list(np.zeros(len(self.counts)))

        for i in range(len(category)):
            for j in range(len(x)):
                #prior = self.counts[i] / sum(self.counts)
                posterior = self.probs[i][j]
                category[i] += -np.log(posterior) * x[j]

        category = list(category)
        print(category)
        arg_max = category.index(max(category))
        return arg_max


#Load data
train_path = r"data\train.npy"
labels_path = r"data\labels.npy"
train = np.load(train_path, allow_pickle = True)[0:800]
labels = np.load(labels_path, allow_pickle = True)[0:800]

test = np.load(train_path, allow_pickle = True)[800:, :-1]
test_labels = np.load(labels_path, allow_pickle = True)[800:]

nb = NaiveBayes(train, labels)
success = 0
count = 0
o = 0
for i in range(len(test)):
    count += 1
    o += nb.predict(test[i])
    if nb.predict(test[i]) == test_labels[i]:
        success += 1
        print("YAY", nb.predict(test[i]))
print(success / count)
print(o)





