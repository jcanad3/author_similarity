from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Input, GlobalAveragePooling1D
from keras.layers import Flatten, Dot
from keras.layers.embeddings import Embedding
from keras.models import Model
from umap import UMAP
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import pickle, random

import warnings
warnings.filterwarnings("ignore")

padded_docs = np.load('../preprocessing/padded_docs.npy')
padded_labels = np.load('../preprocessing/padded_docs_labels.npy')
pair_labels = np.load('../preprocessing/pair_labels.npy')
pair_idx = np.load('../preprocessing/pairs_idx.npy')

embedding_matrix = np.load('../preprocessing/glov_emb_matrix.npy')

print('Padded Docs:', padded_docs.shape)
print('Padded Labels:', padded_labels.shape)
print('Pair Labels:', pair_labels.shape)
print('Pair Idx Matches:', pair_idx.shape)
print('Embedding Matrix:', embedding_matrix.shape)

input_length = 10000
vocab_size = 544698

# sole glov model
input_tens = Input(shape=(input_length,), name='input_1')
x = Embedding(
    vocab_size, 
    100, 
    input_length=input_length, 
    weights=[embedding_matrix],
    trainable=False
)(input_tens)
#x = Flatten()(x)
x = GlobalAveragePooling1D()(x)

model = Model(inputs=input_tens, outputs=x)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.summary()
