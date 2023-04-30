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

nowildfire_path_test = r".\wildfiredata\custom\nowildfire"
wildfire_path_test = r".\wildfiredata\custom\wildfire"

nowildfire_files_test = os.listdir(nowildfire_path_test)
wildfire_files_test = os.listdir(wildfire_path_test)

nowildfire_df_test = pd.DataFrame({'image': [os.path.join(nowildfire_path_test, f) for f in nowildfire_files_test],
                              'label': [0] * len(nowildfire_files_test)})

wildfire_df_test = pd.DataFrame({'image': [os.path.join(wildfire_path_test, f) for f in wildfire_files_test],
                            'label': [1] * len(wildfire_files_test)})

test_data_path = pd.concat([nowildfire_df_test, wildfire_df_test], ignore_index=True)

test_data_path = test_data_path.sample(frac=1)

model_path = "./model/new_model"
predictor = MultiModalPredictor.load(model_path)

if __name__ == '__main__':
    y_pred = predictor.predict(test_data_path)
    y_true = test_data_path['label']
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}")
