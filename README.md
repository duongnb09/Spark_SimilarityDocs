# Spark_SimilarityDocs
Use PySpark to implement the MapReduce algorithm presented in the paper [Pairwise Document Similarity in Large Collections with MapReduce](https://www.aclweb.org/anthology/P/P08/P08-2067.pdf) by Elsayed T., et al. to compute the similarity between two documents. 

Normally, there will be a large number of words existing in a large document collection. However, the most popular words play more important role than the less popular ones. In addition, less words will definitely save some storage spaces and redcue the running time. Thus, the first step is to extract the 1000 most popular words. 
![1000 most popular words](https://github.com/duongnb09/Spark_SimilarityDocs/blob/master/docs/images/word_count.png "Top 1000 words")

Then, compute the inverted indexes which are in the following forms:

*term1: doc1:weight1_1,doc2:weight2_1,doc3:weight3_1,…*

*term2: doc1:weight1_2,doc2:weight2_2,doc3:weight3_2,…*

![Inverted index](https://github.com/duongnb09/Spark_SimilarityDocs/blob/master/docs/images/inverted_index.png "Inverted Index")

The final step is to calculate the similarity
![Similarity matrix](https://github.com/duongnb09/Spark_SimilarityDocs/blob/master/docs/images/similarity_matrix.png "Similarity Matrix")
*di* are documents; *A,B,C a,b,c,f* are words; *a,b,c* are words in the top 1000 word list

**How to run?**

Use *spark-submit*.

For example

*spark-submit --master yarn --num-executors 15 similar_docs.py*

