from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import normalize
import numpy as np
import operator, glob, re

# define 5 documents
#docs = []
#for doc in glob.glob('../../scrape_gutenberg/works/*')[:5]:
#	with open(doc, 'r') as f:
#		text = f.read().replace('\n',' ')
	# remove the links
#	text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
#	text = re.sub('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', '', text, flags=re.MULTILINE)
#	text = re.sub('[0-9]+', '', text)
#	docs.append(text)

#print(docs)
docs = ['Well done Yo Yo!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!']

# create the tokenizer
t = Tokenizer()
# fit the tokenizer on the documents
t.fit_on_texts(docs)
# summarize what was learned
print('Word Index', t.word_index)
words = list(t.word_index.keys())
sorted_words_dict = dict(sorted(t.word_index.items(), key=operator.itemgetter(1)))
print('Sorted Words', list(sorted_words_dict.keys()))
sorted_words = list(sorted_words_dict.keys())
sorted_idx = list(sorted_words_dict.values())

frequency_matrix = t.texts_to_matrix(docs, mode='tfidf')
#frequency_matrix[frequency_matrix == 1] = -1
#frequency_matrix = 1 - frequency_matrix
#frequency_matrix[frequency_matrix == 1] = 0
#frequency_matrix = t.texts_to_matrix(docs, mode='freq')
frequency_matrix = frequency_matrix[:,1:]
frequency_matrix = normalize(frequency_matrix, axis=1, norm='l1')
print(frequency_matrix)

max_len = 4

encoded_docs = []
seq_docs = t.texts_to_sequences(docs)
for idx in range(frequency_matrix.shape[0]):
	if len(seq_docs[idx]) < max_len:
	# pad the sequence
		row_words = [seq_docs[idx]]
		padded_doc = pad_sequences(row_words, maxlen=max_len, padding='post')
		encoded_docs.append(padded_doc[0])
	else:
	# sample from the frequency matrix
		row_weights = frequency_matrix[idx,:]
		# need row indicies not words
		sample = np.random.choice(sorted_idx, size=4, replace=True, p=row_weights)
		encoded_docs.append(sample)

encoded_docs = np.array(encoded_docs)
print('Encoded docs', encoded_docs)
