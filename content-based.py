# Databricks notebook source
#importing libraries
from pyspark.sql.functions import monotonically_increasing_id 
import os
from pyspark.sql.functions import *
from pyspark.sql.types import *
import numpy as np

# COMMAND ----------

# Displaying movies CSV file in data Frame
mov = spark.read.csv('/FileStore/tables/movies.csv', header=True,inferSchema="true")
mov.show()
display(mov)

# COMMAND ----------

# Reading RDD's of tags and removing user Id's Since we don't need user Id wright know
tags = spark.read.csv('/FileStore/tables/tags_l.csv',header=True).drop('userId')
#tags.show()
display(tags)

# COMMAND ----------


#reading the ratings file data and its header which is printed and shown below
#Make sure give the paths with respect to data
read_movies_1m = sc.textFile("/FileStore/tables/movies.csv")
read_movies_1m_header = read_movies_1m.take(1)[0]
read_movies_1m_header

# COMMAND ----------

#Getting total number of umique movies in dataset
s_df = spark.read.format("csv").option("header", "true").load('/FileStore/tables/movies.csv')
a = [i.movieId for i in s_df.select('movieId').distinct().collect()]
print('Number of unique Movies')
len(a)

# COMMAND ----------

#Reading Tag Files 
read_tags_1m = sc.textFile("/FileStore/tables/tags_l.csv")
read_tags_1m_header = read_tags_1m.take(1)[0]
read_tags_1m_header

# COMMAND ----------

#get method for getting a document 
def get_movie_docbyid(movieid):
  return read_tags_1m_data.filter(lambda x: x[0] == movieid).map(lambda x :  [i  for i in x[1]])

# COMMAND ----------

#creation of document for each movie in movies tags
read_tags_1m_data = read_tags_1m.filter(lambda line : line != read_tags_1m_header).map(lambda line:line.split(",")).map(lambda tokens:(int(tokens[1]),tokens[2]))\
.groupByKey().sortByKey()

read_tags_1m_data.take(3)

documents = read_tags_1m_data.map(lambda x :   [i  for i in x[1]])

read_tags = spark.createDataFrame(read_tags_1m_data,('movieid_num', 'document'))
read_tags.show(1572)

# creation of one movie liked by user
#new_movie = read_tags_1m_data.filter(lambda x: x[0] == 1).map(lambda x :  [i  for i in x[1]])
new_movie = get_movie_docbyid(1)

a = [i.movieid_num for i in read_tags.select('movieid_num').distinct().collect()]
#print(len(a))

documents.take(2)
new_movie.take(1)

display(read_tags)

# COMMAND ----------

# the movie Interstellar
print(new_movie.take(1))


# COMMAND ----------

#dataFrame showing a movies, title, documents created from tags
tag_movies = read_tags.join(mov, read_tags.movieid_num == mov.movieId).drop('movieId')
#df_index = df.select("*").withColumn("id", monotonically_increasing_id())
tag_movies_id = tag_movies.select("*").withColumn("id", monotonically_increasing_id())
tag_movies_id.show()
tag_movies.count()


# COMMAND ----------

# Some utility variable 
a.sort()
a.index(109487)
d1 = documents.collect()
d1[826]
a.index(236)

# COMMAND ----------

# computation of bag of words for getting the features
# feature size
bagfwords = read_tags_1m.filter(lambda line : line != read_tags_1m_header).map(lambda line:line.split(",")).map(lambda tokens:((tokens[2]), int(1)))\
.reduceByKey(lambda x,y:x+y)
# bad of worsd
dfbg = spark.createDataFrame(bagfwords,('words','frequency'))
dfbg.show()
display(dfbg)
features_length = bagfwords.count()
print(bagfwords.count())

# COMMAND ----------

#tags count
read_tags_1m_data.count()

# COMMAND ----------

# TFIDF of Documents
from pyspark.mllib.feature import HashingTF, IDF
hashingTF = HashingTF(features_length)
tf = hashingTF.transform(documents)


tf.cache()
idf = IDF().fit(tf)
tfidf = idf.transform(tf)

#tf.cache()
#idf = IDF().fit(tf)
#tfidf = idf.transform(tf)

# COMMAND ----------

tfidf

# COMMAND ----------

#Documents after TFIDF
tfidf
tfidf.take(3)
mtif =  tfidf.map(lambda x : [x])
#mtif.show()
tfdf = spark.createDataFrame(mtif)
tfdf_id = tfdf.select("*").withColumn("id_number", monotonically_increasing_id())

tfdf_id.show()

tag_movies_com = tag_movies_id.join(tfdf_id,tag_movies_id.id == tfdf_id.id_number ).drop('id')

#tag_movies.withColumn("fe",tfdf.select('_1'))
#tfdf.show()

# COMMAND ----------

#Displaying vectors their Documents
tag_movies_com.show()
new_df = tag_movies_com.select("document","_1")
display(new_df)

# COMMAND ----------

#candidate = clean(open('/home/ubuntu/data/essays/candidate').read())
from pyspark.mllib.feature import Normalizer
candidateTf = hashingTF.transform(new_movie)
candidateTfIdf = idf.transform(candidateTf)

"""
def cosine_similarity(candidateTfIdf, Y):
  denom = candidateTfIdf.norm(2) * Y.norm(2)
  if denom == 0.0:
    return -1.0
  else:
    return candidateTfIdf.dot(Y) / float(denom)
"""

#y =   candidateTfIdf.collect()

#ctif =  tfidf.map(lambda x : [x , y])




# COMMAND ----------

# Utility function to compute cosine similarity
from pyspark.mllib.linalg import SparseVector, DenseVector
frequencyDenseVectors_0 = tfidf.map(lambda vector: DenseVector(vector.toArray()))
frequencyDenseVectors_1 =  candidateTfIdf.map(lambda vector: DenseVector(vector.toArray()))

#combfreq = frequencyDenseVectors_0.map(lambda x: x)

# COMMAND ----------

frequencyDenseVectors_1

# COMMAND ----------

y1 =   frequencyDenseVectors_1.collect()

re = frequencyDenseVectors_0.map(lambda x : (x.dot(y1[0]))/(x.norm(2)*y1[0].norm(2)))


# COMMAND ----------

#Displaying similarity score
re.take(10)


# COMMAND ----------

# Utility function for getting movie Ids
#a = [i.movieid for i in tag_movies.select('movieid').collect()]
result1=re.collect()

#exit()
dict1 = {}
list_index  = []
for i in result1:
  if (i > 0.1) :
    #print(result1.index(i))
    dict1[a[result1.index(i)]] = i
    #list_index.append(result1.index(i))
sorted_d = sorted(dict1.items(), key=lambda x: x[1])
top5_r  = sorted_d[-6:]
for i in top5_r:
  list_index.append(i[0])
list_index

# COMMAND ----------

#Movie Ids and their similarity scores
sorted_d[-10:]

# COMMAND ----------

#Data frame showing top movie Id recommendations based with scores
top5_rec = sc.parallelize(top5_r)
df_top5 = spark.createDataFrame([[str(top5_r1[0]), float(top5_r1[1])] for top5_r1 in top5_r ],('movienum','similarity_score'))
df_top5.show()


# COMMAND ----------

# Data frame showing recommended movies, geners title and scores of the movies
comdf_top5 = tag_movies.join(df_top5, df_top5['movienum'] == tag_movies.movieid_num )
#comdf_top5.na.drop()
#comdf_top5.show(1572)
#comdf_top5 = tags.filter( df_top5['_1'] == tags['_c1'] ).show()
comdf_top5.show()


# COMMAND ----------

# Doc2Vec
def doc2vec(movieid):
  new_movie = get_movie_docbyid(movieid)
  candidateTf = hashingTF.transform(new_movie)
  candidateTfIdf = idf.transform(candidateTf)
  return candidateTfIdf


# COMMAND ----------

#cosine similarity
def cosineSimilarity(candidateTfIdf):
  frequencyDenseVectors_1 =  candidateTfIdf.map(lambda vector: DenseVector(vector.toArray()))
  y1 =   frequencyDenseVectors_1.collect()
  re = frequencyDenseVectors_0.map(lambda x : (x.dot(y1[0]))/(x.norm(2)*y1[0].norm(2)))
  return re

# COMMAND ----------

# Get_n _similarities
def get_topn_similarities(n,re,a):
  result1=re.collect()
  dict1 = {}
  list_index  = []
  for i in result1:
    if (i > 0.1)&(i<0.999999) :
      #print(result1.index(i))
      dict1[a[result1.index(i)]] = i
      #list_index.append(result1.index(i))
  sorted_d = sorted(dict1.items(), key=lambda x: x[1])
  top5_r  = sorted_d[-n:]
  for i in top5_r:
    list_index.append(i[0])
  print(list_index)
  return top5_r

# COMMAND ----------

#Data Frame of Similarities
def topn_df(top5_r):
  top5_rec = sc.parallelize(top5_r)
  df_top5 = spark.createDataFrame([[str(top5_r1[0]), float(top5_r1[1])] for top5_r1 in top5_r ],('movienum','similarity_score'))
  return df_top5

# COMMAND ----------


def cont_rec_view(tag_movies,df_top5):
  comdf_top5 = tag_movies.join(df_top5, df_top5['movienum'] == tag_movies.movieid_num )
#comdf_top5.na.drop()
#comdf_top5.show(1572)
#comdf_top5 = tags.filter( df_top5['_1'] == tags['_c1'] ).show()
  comdf_top5.show()
  return comdf_top5
  

# COMMAND ----------

# Case for user had viewed multiple movies
#260,1,16,25,32,335,379,296,858,50
new_user_multiple = [260,1,16,25,32,296,858,50] #new user movie ids (Watched)

#top5_df or dataframe 
schema = StructType([   
                            StructField("movienum", StringType(), True)
                            ,StructField("similarity_score", FloatType(), True)
                        ])
    
similar_movies_df = spark.createDataFrame([], schema)

for movieid in new_user_multiple:
  print(movieid)
  features = doc2vec(movieid)
  cos_smi_fe = cosineSimilarity(features)
  topnsim = get_topn_similarities(5,cos_smi_fe,a)
  similar_movie_df = topn_df(topnsim)
  similar_movies_df = similar_movies_df.union(similar_movie_df)
  
cont_recm = similar_movies_df.orderBy("similarity_score", ascending = False).limit(10)
#find rating send if rating ggreater than 3:




  

# COMMAND ----------

# Showing top 50 similarities
similar_movies_df.show(50)

# COMMAND ----------

#data frame showing top 10 movie id's and scores
cont_recm.show()

# COMMAND ----------

#Joined Data frame that has movies, gneres, title for top 10 recommeded movie ids and scores
df11 = cont_rec_view(tag_movies,cont_recm)

# COMMAND ----------

#s = comdf_top5.select('title').collect()
title = [i.title for i in df11.select('title').collect()]
s_score =  [i.similarity_score for i in df11.select('similarity_score').collect()]

# COMMAND ----------




# COMMAND ----------

# plotting top 10 recommendations

import plotly.graph_objects as go
from plotly.graph_objs import *
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=np.arange(10),
    y=s_score,
    mode="markers+text",
    text= title,
    hovertext=s_score,
    marker=dict(
      size=15,
      color= "blue"
      #set color equal to a variable
     
  )
))


fig.update_layout(title_text="Hover over the points to see the text")

fig.show()

# COMMAND ----------

#Combined dataframe with top 50 similarities
df22 = cont_rec_view(tag_movies,similar_movies_df)
df22.show()

# COMMAND ----------

#utility variables
title1 = [i.title for i in df22.select('title').collect()]
s_score1 =  [i.similarity_score for i in df22.select('similarity_score').collect()]

# COMMAND ----------

# plotting the top 50 movie recommendations
import plotly.graph_objects as go
from plotly.graph_objs import *
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=np.arange(len(title1)),
    y=s_score1,
    mode="markers+text",
    text= title1,
    hovertext=s_score1,
    marker=dict(
      size=15,
      color= "blue"
      #set color equal to a variable
     
  )
))


fig.update_layout(title_text="Hover over the points to see the text")

fig.show()

# COMMAND ----------


