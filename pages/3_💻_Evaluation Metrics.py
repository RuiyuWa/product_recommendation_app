#import packages
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer
from sklearn import datasets
from sklearn.metrics import davies_bouldin_score
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import datasets
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer

#streamlit page
st.set_page_config(page_title="Evaluation Metrics", page_icon="💻")
st.markdown("# Evaluation Metrics")
st.sidebar.header("Evaluation Metrics")
st.write('This page executes the evaluation metrics and illustrate the process in real-time')
st.write('Note: run time is approximately 3 minutes')

####### MAIN SECTION #######

#load data
df = pd.read_csv('product_images.csv')
true_label = pd.read_csv('true_label.csv')

#Standardization and Dimensionality Reduction with PCA
##standardization of data
scaler = StandardScaler()
df_std = scaler.fit_transform(df)


st.header('Method 1: Silhouette Score')
#Choosing number of components
##to keep 90% of variance, 50 components are kept, as shown in the cumulative graph
PC_components = 120
##maximum clusters
cluster_max=20
#Setting number of components
pca=PCA(n_components=PC_components)
##fit the data with selected number of components
pca.fit(df_std)
#calculated resulting components scores
scores_pca=pca.transform(df_std)

# Calculate the Silhouette score for each value of k
silhouette_scores = []
for k in range(2, cluster_max+1):
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = kmeans.fit_predict(scores_pca)
    score = silhouette_score(scores_pca, labels)
    silhouette_scores.append(score)

# Generate the Silhouette score plot
import matplotlib.pyplot as plt
fig6, ax = plt.subplots()
plt.plot(range(2, cluster_max+1), silhouette_scores, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score')
st.pyplot(fig6)

num_cluster_and_silhouette=pd.DataFrame(list(zip(range(2, cluster_max+1),silhouette_scores)), columns=['Number of Clusters (k)','Silhouette Score'])
num_cluster_and_silhouette

st.header('Method 2: Davies-Bouldin Index')
kmeans = KMeans(n_clusters=7, n_init=20, random_state=42)
labels = kmeans.fit_predict(scores_pca)
db_index = davies_bouldin_score(scores_pca, labels)
unique_labels = list(set(labels))
colors = ['red', 'orange', 'grey', 'blue', 'yellow', 'brown', 'green']

results = {}

for i in range(2,cluster_max+1):
    kmeans = KMeans(n_clusters=i, n_init=20, random_state=42)
    labels = kmeans.fit_predict(df_std)
    db_index = davies_bouldin_score(df_std, labels)
    results.update({i: db_index})

fig9, ax = plt.subplots()
plt.plot(list(results.keys()), list(results.values()), marker='o', linestyle='--')
plt.xlabel("Number of clusters")
plt.ylabel("Davies-Boulding Index")
plt.title("Davies-Boulding Index")
st.pyplot(fig9)

num_cluster_and_dbi=pd.DataFrame(list(zip(list(results.keys()),list(results.values()))), columns=['Number of Clusters (k)',' Davies-Boulding Index'])
num_cluster_and_dbi

st.header('Method 3: Adjusted Rand Index')
cluster_num=10
##initialiser and random state as before
kmeans_pca=KMeans(n_clusters=cluster_num, n_init=20, init='k-means++', random_state=42)
kmeans_pca.fit(scores_pca)
#K-means clustering with PCA results
df_pca_kmeans=pd.concat([df.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)
df_pca_kmeans.columns.values[-3:]=['Component 1', 'Component 2', 'Component 3']
#K-means clustering labelsx
df_pca_kmeans['K-means PCA Labels']=kmeans_pca.labels_

##create a copy of df
product_df=df.copy()
##add an index column
product_df.insert(loc=0, column='index', value=range(0, 10000))
##add an column of k-means cluster
product_df.insert(loc=1, column='cluster', value=df_pca_kmeans['K-means PCA Labels'])

##formating input data
true_label.values.ravel()
df_pca_kmeans['K-means PCA Labels'].values.ravel()
##setting input data
labels_true = true_label.values.ravel()
labels_pred = df_pca_kmeans['K-means PCA Labels'].values.ravel()
##adjusted rand index
rand_index_score=metrics.rand_score(labels_true, labels_pred)
rand_index_score