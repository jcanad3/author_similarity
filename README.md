## Author Similarity 
This repository uses word embeddings (GloVe) to compare the similarity of famous authors' works
from the Gutenberg Dataset (https://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html). 

An unsupervised nearest neigbors search on the embeddings is implemented in each of the 
notebooks. Using the generated dataframe,
one could make book recommendations given a large enough dataset.

## Work Similarity
*For the clearest visualization of the results, download visualization/v1_author_pure_glov_vis.ipynb 
or visualization/v2_orig_gutenberg_pure_glov_vis.ipynb to view an interactive plotly 
scatter plot with works labeled by author and title.*

## V1 - First 10,000 
The first 10,000 words of each document are taken as input to the embedding network.

![V1 GloVe Visualization](https://github.com/jcanad3/author_similarity/blob/master/imgs/v1_glov_umap_embeddings.png)

## V2 - L1-TF-IDF Sampling
The tf-idf scores for each word in each document are calculated then L1 normalized, forming a 
probability distribution across the words in the vocabulary for each document. A total of 10,000 samples 
are drawn from each source. If the document has less than 10,000 words, then the sequences are padded 
with zeros to achieve the necessary length.

![V2 GloVe Visualization](https://github.com/jcanad3/author_similarity/blob/master/imgs/v2_glov_umap_embeddings.png)

### V2 Top-1 Matches Sample
A table of the V2 Top-1 matches for a random sample of works:
![Top 1 Matches](https://github.com/jcanad3/author_similarity/blob/master/imgs/v2_random_query_top_1_matches.png)
