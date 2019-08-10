import numpy as np
import pandas as pd
import glob, pickle, os

docs = []
labels = []

individual_labels = []
authors = []
works = []

author_labels_and_keys = {}

label = 0
for d in glob.glob('authors_dirs/*'):
	#print(d)
	author = os.path.basename(d)
	for work in glob.glob(d + '/*'):
		title = os.path.basename(work).replace('.txt', '')
		with open(work, 'r') as f:
			doc = f.read()
		docs.append(doc)
		labels.append(label)

		# create per point data frame
		individual_labels.append(label)
		authors.append(author)
		works.append(title)
		

	author_labels_and_keys[author] = label
	label += 1

# get df for author-label pairs
#data = {'label': list(author_labels_and_keys.values()), 'author': list(author_labels_and_keys.keys())}
#df = pd.DataFrame(data=data)
#df.to_csv('author_label_df.csv', index=False)

# per point df
data = np.column_stack((individual_labels, authors, works))
work_df = pd.DataFrame(data, columns=['author_id_label', 'author', 'work'])
print(work_df)
work_df.to_csv('author_work.csv', index=False) 

#with open('docs_list', 'wb') as f:
#	pickle.dump(docs, f)

#with open('labels', 'wb') as f:
#	pickle.dump(labels, f)
