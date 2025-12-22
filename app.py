import streamlit as st
from PIL import Image
from main import get_similar_images



st.set_page_config(page_title="Similar Image Search", layout="wide")

st.title("🍎 Image Similarity Search")
st.write("Upload an image to find similar items in our database.")





uploaded_file = st.file_uploader('Uplode Image' , type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    query_image = Image.open(uploaded_file)
    st.image(query_image, caption="Your Uploaded Image", width=300)

    if st.button("🚀 Find Similar Images"):
        
        img_path , scores  = get_similar_images(query_image)

        st.divider()
        st.subheader("Top Similar Results")
        
        # Create a grid of 5 columns for the results
        cols = st.columns(len(img_path))
        
        for i , (image, score) in enumerate(zip(img_path , scores)):
            with cols[i]:
                # Display the image fetched from the HF stream
                image = Image.open(image)
                st.image(image, use_container_width=True)
                
                # Show the similarity score and index
                st.metric(label="Similarity Score", value=f"{score*100:.2f}")
                #st.caption(f"Dataset Index: {res['index']}")


st.divider()
st.markdown("""
### 📚 References & Credits
---
* **📊 Dataset:** [ImageNet 1000 (Mini) from Kaggle](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000)
* **🧠 Notebook Inspiration:** [Image Similarity Search using VGG16 + Cosine](https://www.kaggle.com/code/ashiskumarbera/image-similarity-search-using-vgg16-cosine)
* **🛠️ Tech Stack:** Built with **PyTorch**, **Streamlit**, and **PIL**
""")








