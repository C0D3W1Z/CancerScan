import os
import pandas as pd
import warnings
import autogluon
warnings.filterwarnings('ignore')
from autogluon.multimodal import MultiModalPredictor
from autogluon.core.utils.loaders import load_pd
from autogluon.tabular import TabularDataset
from sklearn.metrics import accuracy_score
import uuid

image_path = './img.jpg'

model_path = "./model/skincancermodel"
predictor = MultiModalPredictor.load(model_path)

if __name__ == '__main__':
    predictions = predictor.predict({'image': [image_path]})
    proba = predictor.predict_proba({'image': [image_path]})
    print(proba)
    print(predictions)
