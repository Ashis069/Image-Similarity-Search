import torch
import torch.nn.functional as F
import streamlit as st
from models import FeatureExtractor
from config import transform


@st.cache_resource
def get_data_model():
    features = FeatureExtractor()

    data = features.get_feture_bank("./stubs/data_and_features.pt")

    return features , data



def load_image(img):
    tensor= transform(img).unsqueeze(0) # -> (1,h,w,c)
    return tensor
    


def get_similar_images(query_image , k = 5 ):

    features , data = get_data_model()


    img_bank = data['images']
    feat_bank = data['features']    


    query_features = features.extract_features(load_image(query_image))

    feature_bank = feat_bank.to(query_features.device)

    # apply cosine similarity 
    output = F.cosine_similarity(query_features, feature_bank)
    
    top_k = torch.topk(output, k=5)

    Image_Indices = top_k.indices.tolist()
    Similarity_Scores =  top_k.values.tolist()

    # fix the path 
    local_root = "./datasets/imagenet-mini/" 
    
    kaggle_prefix = "/kaggle/input/imagenetmini-1000/imagenet-mini/"

    path = [img_bank[i].replace(kaggle_prefix, local_root) for i in Image_Indices]

    return path , Similarity_Scores

