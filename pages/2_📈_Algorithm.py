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
st.set_page_config(page_title="The Algorithm", page_icon="📈")
st.sidebar.header("The Algorithm")
st.title('The Algorithm')
st.write('This page executes the full K-means clustering algorithm and illustrate the process in real-time')
st.write('Note: run time is approximately 4 minutes')

####### MAIN SECTION #######

#load data
df = pd.read_csv('product_images.csv')
true_label = pd.read_csv('true_label.csv')


st.header('1. Dimensionality Reduction with Principle Component Analysis')
#Standardization and Dimensionality Reduction with PCA
##standardization of data
scaler = StandardScaler()
df_std = scaler.fit_transform(df)
##fit standardized data with PCA
pca = PCA()
pca.fit(df_std)
#PCA Explained Variance Ratio
## pca.explained_variance_ratio_ shows how much variance is explained in each principle component
##plot the explained variance ratio for each principal component
fig1, ax = plt.subplots()
plt.plot(range(1, 785), pca.explained_variance_ratio_)
plt.title('Explained Variance by Components')
plt.xlabel('Number of Principal Component')
plt.ylabel('Explained Variance Ratio')
st.pyplot(fig1)

##plot the cumulative explained variance 
fig2, ax = plt.subplots()
plt.plot(range(1, 785), pca.explained_variance_ratio_.cumsum())
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.vlines(x=120, ymin=0.2, ymax=0.9, colors='r',linestyles='dashed')
plt.hlines(y=0.9, xmin=0.2, xmax=120, colors='r',linestyles='dashed')
st.pyplot(fig2)
st.write("The cumulative explained variance plot suggests 120 principle components to be included to reflect 90% of the variance")

st.header('2. Determining Number of Clusters')
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

st.subheader('Method 1: Distortion Score Elbow for Kmeans Clustering')
# Instantiate the clustering model and visualizer
km = KMeans(n_init=20, random_state=42)

fig3, ax = plt.subplots()
visualizer = KElbowVisualizer(km, k=(2,cluster_max+1))
visualizer.fit(scores_pca)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure
st.pyplot(fig3)

st.subheader('Method 2: Gap Statistics for Kmeans Clustering')
X=df_std

def optimalK(data, nrefs=3, maxClusters=20):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
    for gap_index, k in enumerate(range(1, maxClusters)):
        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)
        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)
            
            # Fit to it
            km = KMeans(k, n_init=20)
            km.fit(randomReference)
            
            refDisp = km.inertia_
            refDisps[i] = refDisp
        # Fit cluster to original data and create dispersion
        km = KMeans(k, n_init=20)
        km.fit(data)
        
        origDisp = km.inertia_
        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)
        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap
        
        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)
    return (gaps.argmax() + 1, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal
# Automatically output the number of clusters
k, gapdf = optimalK(X, nrefs=3, maxClusters=11)
print('Optimal k is: ', k)
# Visualization
fig4, ax = plt.subplots()
plt.plot(gapdf.clusterCount, gapdf.gap, linewidth=3)
plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')
plt.grid(True)
plt.xlabel('Cluster Count')
plt.ylabel('Gap Value')
plt.title('Gap Values by Cluster Count')
st.pyplot(fig4)

st.header('3. K-means Clustering Algorithm')
#K-means clustering
cluster_num=10
##initialiser and random state as before
kmeans_pca=KMeans(n_clusters=cluster_num, n_init=20, init='k-means++', random_state=42)
kmeans_pca.fit(scores_pca)
#K-means clustering with PCA results
df_pca_kmeans=pd.concat([df.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)
df_pca_kmeans.columns.values[-3:]=['Component 1', 'Component 2', 'Component 3']
#K-means clustering labelsx
df_pca_kmeans['K-means PCA Labels']=kmeans_pca.labels_
df_pca_kmeans.head()
# Plot the clustered data
fig5, ax = plt.subplots()
plt.scatter(scores_pca[:, 0], scores_pca[:, 1], c=kmeans_pca.labels_, cmap=plt.cm.viridis, s=3)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Clusters by PCA Components')
plt.colorbar(label='Cluster')
st.pyplot(fig5)

st.subheader('Clustered Data')
##create a copy of df
product_df=df.copy()
##add an index column
product_df.insert(loc=0, column='index', value=range(0, 10000))
##add an column of k-means cluster
product_df.insert(loc=1, column='cluster', value=df_pca_kmeans['K-means PCA Labels'])
product_df

