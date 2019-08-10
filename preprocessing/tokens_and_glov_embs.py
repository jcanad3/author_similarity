from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle, random

def gen_pairs(labels):
	# make data set with padded doc sequences
	labels = np.array(labels)
	uniq_labels = np.unique(labels)
	
	pairs = []
	pair_labels = []
	for label in uniq_labels:
		match_idx = np.argwhere(labels == label)
	
		non_match_idx = np.argwhere(labels != label).flatten()
		matches = match_idx.tolist()
		for x in matches:
			x_1_idx = x[0]
			for point in matches:
				# pos match
				x_2_idx = point[0]
				if x_1_idx != x_2_idx and ((x_1_idx, x_2_idx) not in pairs or (x_2_idx, x_1_idx) not in pairs):
					if random.uniform(0,1) < 0.5:
						pairs.append([x_1_idx, x_2_idx])
					else:
						pairs.append([x_2_idx, x_1_idx])
					pair_labels.append(1)
	
					# when pos match occurs, gen a neg match
					while True:
						x_3_idx = np.random.choice(non_match_idx, 1)[0]
						if x_1_idx != x_3_idx and ((x_1_idx, x_3_idx) not in pairs or (x_3_idx, x_1_idx) not in pairs):
							break
					if random.uniform(0,1) < 0.5:
						pairs.append([x_1_idx, x_3_idx])
					else:
						pairs.append([x_3_idx, x_1_idx])
					pair_labels.append(0)
		print(pairs)	
		
	print(pairs)
	
	pairs_idx = np.array(pairs)
	pair_labels = np.array(pair_labels)
	np.save('pairs_idx.npy', pairs_idx)
	np.save('pair_labels.npy', pair_labels)
	
	print('Pairs shape:', pairs_idx.shape)
	print('Pairs Labels:', pair_labels.shape)

def gen_padded_docs(docs):
	# fit tokens to words
	t = Tokenizer()
	t.fit_on_texts(docs)
	vocab_size = len(t.word_index) + 1
	encoded_docs = t.texts_to_sequences(docs)
#	print(encoded_docs)
	
	# pad docs
	max_length = 10000
	input_length = 10000
	padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
	np.save('padded_docs.npy', padded_docs)
#	print(padded_docs)

	return t, vocab_size

def gen_embedding_matrix(t, vocab_size):
	# get embedding matrix
	embeddings_index = dict()
	f = open('glove_embs/glove.6B.100d.txt')
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	
	embedding_matrix = np.zeros((vocab_size, 100))
	for word, i in t.word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
	
	np.save('glov_emb_matrix.npy', embedding_matrix)	

# docs 
with open('docs_list.data', 'rb') as f:
	docs = pickle.load(f)

# labels 
labels = np.load('docs_labels.npy')

#gen_pairs(labels)

t, vocab_size = gen_padded_docs(docs)
gen_embedding_matrix(t, vocab_size)
