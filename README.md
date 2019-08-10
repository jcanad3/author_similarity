## Author Similarity 
This repository uses word embeddings (GloVe) to compare the similarity of famous authors' works
from the Gutenberg Dataset (https://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html). 
This approach captures both the style (word choice) and the topic which the work explores.  

An unsupervised nearest neigbors search on the embeddings is implemented in the 
visualization/author_pure_glov_vis.ipynb notebook. Using the generated dataframe,
one could make book recommendations given a large enough dataset.

![GloVe Visualization](https://github.com/jcanad3/author_similarity/blob/master/imgs/glov_umap_embeddings.png)

## Work Similarity
Download the visualization/author_pure_glov_vis.ipynb to view an interactive plotly scatter plot 
with works labeled by author and title. 
