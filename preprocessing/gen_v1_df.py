import pandas as pd
import glob, os

author_df = pd.DataFrame()
count = 0
#for author_dir in glob.glob('../authors_dirs/*'):
#	author = os.path.basename(author_dir)
#
#	for work_path in glob.glob(author_dir + '/*'):
#		work = os.path.basename(work_path)
#		print(work)
#
#		data = {'work': [work], 'author': [author], 'work_id': [count]}
#		partial_df = pd.DataFrame(data=data)
#		partial_df = partial_df.reindex(columns=['work_id', 'author', 'work'])
#
#		author_df = author_df.append(partial_df, ignore_index=True)
#		count += 1

for work_path in glob.glob('../data/Gutenberg/txt/*'):
	work = os.path.basename(work_path)
	author = work.split('___')[0]
	author = author.replace(' ', '_')
	work = work.split('___')[1]
	work = work.replace(' ', '_')
	data = {'work': [work], 'author': [author], 'work_id': [count]}
	partial_df = pd.DataFrame(data=data)
	partial_df = partial_df.reindex(columns=['work_id', 'author', 'work'])

	author_df = author_df.append(partial_df, ignore_index=True)
	count += 1


author_df.to_csv('v1_author_df.csv', index=False) 
