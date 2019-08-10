from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Input, GlobalAveragePooling1D
from keras.layers import Flatten, Dot
from keras.layers.embeddings import Embedding
from keras.models import Model
import numpy as np
import pickle


def emb_layer(input_tens, embedding_matrix, input_length):
	x = Embedding(
		vocab_size, 
		100, 
		input_length=input_length, 
		weights=[embedding_matrix],
		trainable=False
	)(input_tens)
	x = GlobalAveragePooling1D()(x)
	#x = Flatten()(x)
	x = Dense(100, activation='linear')(x)

	return x

padded_docs = np.load('padded_docs.npy')
pair_labels = np.load('pair_labels.npy')
pair_idx = np.load('pairs_idx.npy')

embedding_matrix = np.load('glov_emb_matrix.npy')

input_length = 10000
vocab_size = 544698

input_tens_1 = Input(shape=(input_length,))
input_tens_2 = Input(shape=(input_length,))
emb_1 = emb_layer(input_tens_1, embedding_matrix, input_length)
emb_2 = emb_layer(input_tens_2, embedding_matrix, input_length)

merged = Dot(normalize=True, axes=1)([emb_1, emb_2])

out = Dense(1, activation='sigmoid')(merged)

model = Model(inputs=[input_tens_1, input_tens_2], outputs=out)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# feed data in global batches
for epoch in range(100):
	print('---------------------------------' + str(epoch) + '---------------------------------')
	for batch in range(0, pair_labels.shape[0], 25000):
		x_1 = padded_docs[pair_idx[batch:batch+25000,0], :]
		x_2 = padded_docs[pair_idx[batch:batch+25000,1], :]
		y = pair_labels[batch:batch+25000]

		print('X_1 Shape', x_1.shape)
		print('X_2 Shape', x_2.shape)
		print('Y Shape', y.shape) 
		model.fit(
			[x_1,x_2], 
			y, 
			shuffle=True, 
			batch_size=128,
			epochs=1, 
			validation_split=0.2
		)
	model.save_weights('ckpt_weights/model_weights_e_' + str(epoch))
