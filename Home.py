import streamlit as st

#streamlit page
st.set_page_config(page_title="Home", page_icon="🏠")
st.sidebar.success("Select a section above.")
st.title('Product Recommendation App')
st.write("👋 Welcome!")
st.write("👠 This app  uses K-means clustering algorithm to provide recommendations for clothing!")
st.write("👈 Select a section from the sidebar to get started!")


from PIL import Image

image = Image.open('cover_image.jpeg')

st.image(image)

