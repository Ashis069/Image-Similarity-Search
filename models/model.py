import torch 
import torch.nn as nn
import sys
import os
sys.path.append("../")
from config import device
from torch.utils.data import DataLoader
from torchvision import models , datasets
from config import transform , device 
from tqdm import tqdm


class FeatureExtractor():
        def __init__(self):
             self.model = self._load_model()

        def _load_model(self):
             vgg = models.vgg16(weights=None)
             state_dict = torch.load(
                  "./weights/vgg16-397923af.pth",
                  map_location="cpu"
                  )
             vgg.load_state_dict(state_dict)
             model = nn.Sequential(*list(vgg.features.children()))
             return model.to(device)
        
        
        def extract_features(self,img_tensor):
            
            self.model.eval().to(device)
            with torch.no_grad():
                features = self.model(img_tensor.to(device))   
            features = torch.flatten(features,start_dim=1)
            
            return features
        
        
        def get_feture_bank(self,stub_path):

            if  stub_path  and os.path.exists(stub_path) :
                data = torch.load(
                     stub_path,
                     map_location="cpu")
                return data
            
            dataset = datasets.ImageFolder(
                 root='/datasets/imagenet-mini/train',
                 transform=transform
                 )   
            
            dataloader = DataLoader(
                 dataset,
                 batch_size=32,
                 shuffle=False
                 )
            
            feature_bank = []

          
            for i , (images ,_) in tqdm(enumerate(dataloader)):
                images = images.to(device)

                features = self.extract_features(images)
                feature_bank.append(features.cpu())



            feature_bank = torch.cat(feature_bank, dim=0)

            image_list = [path for path, label in dataset.samples]


            data = {'images': image_list,
                    'features': feature_bank}
            
            if stub_path :
                 torch.save(data,stub_path)

            return data
        
        
        



