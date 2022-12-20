import streamlit as st
import matplotlib.image as mimg
import matplotlib.pyplot as plt
from PIL import Image


st.title('Music Generation App!')

st.header("Abhishek Nair  \n")
st.subheader("Cascabel Curve Cohort")

img = mimg.imread('./data/images/main.jpeg')

st.image(img, width=700)

st.markdown("""This is a web application to create music \
**For all. Musical or not** """)





