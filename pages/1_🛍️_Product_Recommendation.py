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
st.set_page_config(page_title="Product Recommendation", page_icon="🛍️")
st.markdown("# Product Recommendation")
st.sidebar.header("Product Recommendation")
st.write('This page recommends products based on clustering algorithm')
st.write('Click the buttons to get started')

####### MAIN SECTION #######

#load data
df = pd.read_csv('product_images.csv')
true_label = pd.read_csv('true_label.csv')


#Standardization and Dimensionality Reduction with PCA
##standardization of data
scaler = StandardScaler()
df_std = scaler.fit_transform(df)
##fit standardized data with PCA
pca = PCA()
pca.fit(df_std)
#Choosing number of components
##to keep 90% of variance, 120 components are kept, as shown in the cumulative graph
PC_components = 120
##maximum clusters
cluster_max=20
#Setting number of components
pca=PCA(n_components=PC_components)
##fit the data with selected number of components
pca.fit(df_std)
#calculated resulting components scores
scores_pca=pca.transform(df_std)

#setting number of clusters
cluster_num=10
#K-means clustering
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


# Select 10 random rows from the dataframe
product_df_random = product_df.sample(10)
# Get the pixel data for the selected rows
random_pixels = product_df_random.loc[:, 'pixel1':'pixel784'].values
# Display the selected pixels
print(random_pixels)


image_data = np.asarray(random_pixels.reshape(-1, 28, 28))

for i in range(10):
    img = Image.fromarray(np.uint8(image_data[i]))
    st.image(img, caption='Product {}'.format(i+1))
    if st.button('Product {}'.format(i+1)):
        # Get the cluster label of the random item
        cluster_label = product_df_random['cluster'].values[0]
        # Filter the dataframe to get all items belonging to that cluster
        cluster_items = product_df[product_df['cluster'] == cluster_label]
        
        # Randomly select 4 images from the same cluster
        random_images = cluster_items.sample(4)

        # Display the 4 random images
        for j in range(4):
            rec_random_pixels = random_images.loc[:, 'pixel1':'pixel784'].values
            rec_image_data = np.asarray(rec_random_pixels.reshape(-1, 28, 28))
            rec_img = Image.fromarray(np.uint8(rec_image_data[j]))
            st.image(rec_img, caption='Random image {}'.format(j+1))
                         




















