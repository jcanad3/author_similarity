from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
import pickle, random, glob, re, os, operator

import warnings
warnings.filterwarnings("ignore")

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

def fit_tokenizer(docs):
	t = Tokenizer()
	t.fit_on_texts(docs)
	vocab_size = len(t.word_index) + 1

	with open('tokenizer.pkl', 'wb') as f:
		pickle.dump(t, f)

	return t, vocab_size

def gen_padded_docs(t, docs):
	# fit tokens to words
	padded_docs = []
	#words = list(t.word_index.keys())
	#sorted_words_dict = dict(sorted(t.word_index.items(), key=operator.itemgetter(1)))
	sorted_idx = list(t.word_index.values())
	#sorted_idx = list(sorted_words_dict.values())

	# text to sequence
	seq_docs = t.texts_to_sequences(docs)

	# integer encode documents
	frequency_matrix = t.texts_to_matrix(docs, mode='tfidf')
	frequency_matrix = frequency_matrix[:,1:]
	frequency_matrix = normalize(frequency_matrix, axis=1, norm='l1')

	max_len = 10000
	encoded_docs = []
	for idx in range(frequency_matrix.shape[0]):
		if len(seq_docs[idx]) < max_len:
			# pad the sequence
			row_words = [seq_docs[idx]]
			padded_doc = pad_sequences(row_words, maxlen=max_len, padding='post')
			encoded_docs.append(padded_doc[0])
		# sample from the frequency matrix
		else:
			row_weights = frequency_matrix[idx,:]
			# need row indicies not words
			sample = np.random.choice(sorted_idx, size=max_len, replace=True, p=row_weights)
			encoded_docs.append(sample)
	
	padded_docs = np.array(encoded_docs)	
#	np.save('v2_padded_docs.npy', padded_docs)
#	print('Padded Docs Shape', padded_docs.shape)
#	print(padded_docs)

	return padded_docs

def get_docs(docs_dir, t, start_idx, end_idx):
	num_files_in_dir = len(glob.glob(docs_dir + '*')) + 1
	if end_idx > num_files_in_dir:
		end_idx = num_files_in_dir


	docs = []
	for doc in glob.glob(docs_dir + '*')[start_idx:end_idx]:
		doc_path = os.path.basename(doc)
		with open(doc, 'r') as f:
			text = f.read().replace('\n',' ')
		# remove the links
		text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
		text = re.sub('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', '', text, flags=re.MULTILINE)
		text = re.sub('[0-9]+', '', text)
		if 'START OF THIS PROJECT GUTENBERG' in text:
			text = text.split('START OF THIS PROJECT GUTENBERG')[1]
		elif 'START OF THE PROJECT GUTENBERG' in text:
			text = text.split('START OF THE PROJECT GUTENBERG')[1]
		else:
			pass		

		if 'End of Project Gutenberg' in text:
			text = text.split('End of Project Gutenberg')[0]
		elif 'End of the Project Gutenberg' in text: 
			text = text.split('End of the Project Gutenberg')[0]
		elif 'End of this Project Gutenberg' in text:
			text = text.split('End of this Project Gutenberg')[0]
		elif 'END OF THIS PROJECT GUTENBERG' in text:
			text = text.split('END OF THIS PROJECT GUTENBERG')[0]
		elif 'END OF THE PROJECT GUTENBERG' in text:
			text = text.split('END OF THE PROJECT GUTENBERG')[0]
		elif 'END OF THIS PROJECT GUTENBERG EBOOK' in text:
			text = text.split('END OF THIS PROJECT GUTENBERG EBOOK')[0]
		else:
			pass
		docs.append(text)

	padded_docs = gen_padded_docs(t, docs)

	return padded_docs

def gen_embedding_matrix(t, vocab_size):
	# get embedding matrix
	embeddings_index = dict()
	f = open('../glove_embs/glove.6B.100d.txt')
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

def gen_author_df():
	author_df = pd.DataFrame()
	doc_ids = []
	doc_id = 0
	for doc in glob.glob('../../scrape_gutenberg/works/*'):
		doc_path = os.path.basename(doc)
		doc_ids = [doc_id]
		authors = [doc_path.split('---')[1].replace('.txt', '')]
		works = [doc_path.split('---')[0]]
	
		data = {'doc_id': doc_ids, 'author': authors, 'work': works}
		partial_df = pd.DataFrame(data=data)
		partial_df = partial_df.reindex(columns=['doc_id', 'author', 'work'])
		author_df = author_df.append(partial_df, ignore_index=True)
		doc_ids.append(doc_id)
		doc_id += 1

	author_df.to_csv('v2_author_df.csv', index=False)
	np.save('v2_padded_ids.npy', doc_ids)

#gen_author_df()
#docs = []
#for doc in glob.glob('../../scrape_gutenberg/works/*')[:1000]:
#	doc_path = os.path.basename(doc)
#	with open(doc, 'r') as f:
#		text = f.read().replace('\n',' ')
#	# remove the links
#	text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
#	text = re.sub('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', '', text, flags=re.MULTILINE)
#	text = re.sub('[0-9]+', '', text)
#	docs.append(text)

#t, vocab_size = fit_tokenizer(docs)
#print('Fit Tokenizer.')
#print('Vocab Size', vocab_size)
#gen_embedding_matrix(t, vocab_size)
#print('Generated Embedding Matrix.')

if __name__ == '__main__':
	with open('../preprocessing/tokenizer.pkl', 'rb') as f:
		t = pickle.load(f)
	
	padded_docs = np.asarray(())
	for d in range(0, 10, 5):
		docs_sample = get_docs(t, d, d+5)
		if padded_docs.size == 0:
			padded_docs = docs_sample
		else:
			padded_docs = np.vstack((padded_docs, docs_sample))
	
		print('Padded Docs', padded_docs.shape)
	
	print('Padded Docs', padded_docs.shape)
	print(padded_docs)
