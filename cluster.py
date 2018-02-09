import json
import string
import collections
import re
import numpy as np
import sys
 
import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint

from scipy.spatial.distance import cosine

from scipy.cluster.vq import kmeans,vq

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

file_name = sys.argv[1]
num_cluster = int(sys.argv[2])

print("Reading")
with open(file_name) as f:
    data=json.load(f)

print("Done reading")
def process_text(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def cluster_texts(texts, clusters=3):
    """ Transform texts to Tf-Idf coordinates and cluster texts using K-Means """
    vectorizer = TfidfVectorizer(tokenizer=process_text,
                                 stop_words=stopwords.words('english'),
                                 lowercase=True)
 
    tfidf_model = vectorizer.fit_transform(texts)
    print("*******************************")
    # print(np.array(tfidf_model[0].todense())[0])
    km_model = KMeans(n_clusters=clusters)
    km_model.fit(tfidf_model)
 
    clustering = collections.defaultdict(list)
 
    for idx, label in enumerate(km_model.labels_):
        clustering[label].append(idx)

    print(clustering)

    centroids = km_model.cluster_centers_

    centers = []
    for i in range(len(centroids)):
        centroid = centroids[i]
        min_dist = sys.maxint
        min_dist_sent = -1
        centroid_dist = []
        for sent in clustering[i]:
            # print(tfidf_model[sent].todense())
            # print(centroid)
            dist = cosine(np.array(tfidf_model[sent].todense())[0], centroid)
            centroid_dist.append([sent, dist])
        centroid_dist = sorted(centroid_dist, key=lambda item: item[1])
        centers.append([item[0] for item in centroid_dist[:min(5, len(centroid_dist))]])

    print(centroids)
    print(centers)
 
    return clustering,centers

clusters = []
cnt = 0
for item in data:
    print(cnt, len(data))
    cnt += 1
    clus = {}
    clus['complex'] = item['complex']
    clus['gold'] = item['gold']
    clus['candidates'] = {}
    cand = item['candidates']
    sent = [c['sent'] for c in cand]
    cluster,centers = cluster_texts(sent, num_cluster)
    clus['centers'] = centers
    for k in cluster.keys():
        clus['candidates'][str(k)] = [{'val':cand[p], 'original_rank': p} for p in cluster[k]]
    clusters.append(clus)

with open("cluster_seq2seq_att_" + str(num_cluster) + ".json", "w") as f:
    json.dump(clusters, f, sort_keys=True, indent=4)


