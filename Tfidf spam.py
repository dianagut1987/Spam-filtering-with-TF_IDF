#!/usr/bin/env python
# coding: utf-8

# In[72]:



import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer

#docs=[" Important Career Center Information A new spin on a recent trend to spam MIT people offering them job services.",
       #"It's a spam message"]
#docs=["we offering Traffic services",
#       "BOOST Your Website's Traffic An ad for cheesy-sounding internet services."]
docs=["Rick Wilson Tweeted: Now you know why he doesn’t want them to testify",
      " Get Rich Click Another lame get rich quick scam"]
 
#instantiate CountVectorizer()
cv=CountVectorizer()
 
# this steps generates word counts for the words in docs
word_count_vector=cv.fit_transform(docs)

#how much docs in rows and columns
word_count_vector.shape
 
#compute Idf
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)
 
# print idf values
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])
 
# sort ascending
df_idf.sort_values(by=['idf_weights'])

# get the first vector out (for the first document)
first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[0]
 
# place tf-idf values in a pandas data frame
df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False)
 
#compute the tf-idf scores for any document or set of documents
#count matrix
count_vector=cv.transform(docs) 
# tf-idf scores
tf_idf_vector=tfidf_transformer.transform(count_vector)
#print(tf_idf_vector) 
    
#settings for count vectorizer
tfidf_vectorizer=TfidfVectorizer(use_idf=True)
tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(docs)
 
#get the tf-idf scores of a set of documents.
fitted_vectorizer=tfidf_vectorizer.fit(docs)
tfidf_vectorizer_vectors=fitted_vectorizer.transform(docs) 

print("TF IDF values ")
print(tfidf_vectorizer_vectors)
print("\n")
print(tfidf_vectorizer.vocabulary_)
print("\n")

#correlation matrix between the documents 
vecs = tfidf_vectorizer.fit_transform(docs)
matrix = ((vecs * vecs.T).A)
print(matrix)
print("\n")

#defining each row as document
#row1=matrix[0]
#row2=matrix[1]
# row3=matrix[2]
# row4=matrix[3]
# row5=matrix[4]

#calculate distance beetween documents with cosine 
#result0=1-spatial.distance.cosine(row1,row2)
# result1=1-spatial.distance.cosine(row1,row2)
# result2=1-spatial.distance.cosine(row1,row3)
# result3=1-spatial.distance.cosine(row1,row4)
# result3=1-spatial.distance.cosine(row1,row5)
# result4=1-spatial.distance.cosine(row2,row3)
# result5=1-spatial.distance.cosine(row2,row4)
# result6=1-spatial.distance.cosine(row2,row5)
# result7=1-spatial.distance.cosine(row3,row4)
# result7=1-spatial.distance.cosine(row3,row5)
# result8=1-spatial.distance.cosine(row4,row5)


# #similarity to each one of the document
# print(matrix.mean(axis=1))
# print("\n")
# #similarity between all the documents
# print(matrix.mean())
# print("\n")

#differnces between each 2 documents
#print("similarity between first and second",result1)
# print("similarity between first and third",result2)
# print("similarity between first and forth",result3)
# print("similarity between first and fifth",result4)
# print("similarity between second and third",result5)
# print("similarity between second and forth",result6)
# print("similarity between third and forth",result6)
# print("similarity between third and fifth ",result7)
# print("similarity between forth and fifth ",result8)






