import re
import numpy as np


def tokenize(text):
    # obtains tokens with a least 1 alphabet
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())

def mapping(tokens):
    print(set(tokens))
    l = len(list(set(tokens)))
    print(l)
    word_to_id = dict()
    id_to_word = dict()

    for i, token in enumerate(set(tokens)):
        word_to_id[token] = i
        id_to_word[i] = token

    return word_to_id, id_to_word

def generate_training_data(tokens, word_to_id, window_size):
    N = len(tokens)
    X, Y = [], []

    for i in range(N):
        nbr_inds = list(range(max(0, i - window_size), i)) + \
                   list(range(i + 1, min(N, i + window_size + 1)))
        print(nbr_inds)
        for j in nbr_inds:
            X.append(word_to_id[tokens[i]])
            Y.append(word_to_id[tokens[j]])

    X = np.array(X)
    X = np.expand_dims(X, axis=0)
    Y = np.array(Y)
    Y = np.expand_dims(Y, axis=0)

    return X, Y

doc = "After the deduction of the costs of investing, " \
      "beating the stock market is a loser's game."
tokens = tokenize(doc)
word_to_id, id_to_word = mapping(tokens)
X, Y = generate_training_data(tokens, word_to_id, 3)
print("this is train X and Y")
print(X)
print(Y)
vocab_size = len(id_to_word)
m = Y.shape[1]
# turn Y into one hot encoding
Y_one_hot = np.zeros((vocab_size, m))
print(Y_one_hot.shape,"the shape of Y_one_hot is")
Y_one_hot[Y.flatten(), np.arange(m)] = 1
for i in range(Y_one_hot.shape[1]):
    print(Y_one_hot[:,i])

parameters = initialize_parameters(13,5)

batch_inds = list(range(0,m,batch_size))
print(batch_inds)
np.random.shuffle(batch_inds)
print(batch_inds)
for i in batch_inds:
    X_batch = X[:,i:i+batch_size]
    Y_batch = Y_one_hot[:,i:i+batch_size]

print("batch_size")
print(X_batch)
print(Y_batch)

