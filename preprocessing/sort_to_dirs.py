from shutil import copyfile
import glob, os

author_dirs = []
count = 0 
for f in glob.glob('Gutenberg/txt/*'):
	print(f)
	author_dir = os.path.basename(f).split('_')[0].replace(' ', '_')
	work = os.path.basename(f).split('___')[1].replace(' ', '_')
	#print(work)
	#os.system('mv ' + f + ' authors_dirs/' + author_dir + '/' + work)
	copyfile(f, 'authors_dirs/' + author_dir + '/' + work)
	#if author_dir not in author_dirs:
	#	author_dirs.append(author_dir)

#for d in author_dirs:
#	os.makedirs('authors_dirs/' + d, exist_ok=True)
