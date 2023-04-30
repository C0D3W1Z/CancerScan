import os
import pandas as pd
import warnings
import autogluon
warnings.filterwarnings('ignore')
from autogluon.multimodal import MultiModalPredictor
from autogluon.core.utils.loaders import load_pd
from autogluon.tabular import TabularDataset

akiec_path = r".\input\akiec"
bcc_path = r".\input\bcc"
bkl_path = r".\input\bkl"
df_path = r".\input\df"
mel_path = r".\input\mel"
nv_path = r".\input\nv"
vasc_path = r".\input\vasc"

akiec_files = os.listdir(akiec_path)
bcc_files = os.listdir(bcc_path)
bkl_files = os.listdir(bkl_path)
df_files = os.listdir(df_path)
mel_files = os.listdir(mel_path)
nv_files = os.listdir(nv_path)
vasc_files = os.listdir(vasc_path)

akiec_df = pd.DataFrame({'image': [os.path.join(akiec_path, f) for f in akiec_files],
                              'label': [0] * len(akiec_files)})

bcc_df = pd.DataFrame({'image': [os.path.join(bcc_path, f) for f in bcc_files],
                            'label': [1] * len(bcc_files)})

bkl_df = pd.DataFrame({'image': [os.path.join(bkl_path, f) for f in bkl_files],
                              'label': [2] * len(bkl_files)})

df_df = pd.DataFrame({'image': [os.path.join(df_path, f) for f in df_files],
                            'label': [3] * len(df_files)})

nv_df = pd.DataFrame({'image': [os.path.join(nv_path, f) for f in nv_files],
                            'label': [4] * len(nv_files)})

vasc = pd.DataFrame({'image': [os.path.join(vasc_path, f) for f in vasc_files],
                              'label': [5] * len(vasc_files)})

mel_df = pd.DataFrame({'image': [os.path.join(mel_path, f) for f in mel_files],
                              'label': [6] * len(mel_files)})

train_data_path = pd.concat([akiec_df, bcc_df, bkl_df, df_df, nv_df, vasc, mel_df], ignore_index=True)

model_path = "./model/new_model_new"

predictor = MultiModalPredictor(label="label", path=model_path)

if __name__ == '__main__':
    predictor.fit(
    train_data=train_data_path,
    time_limit=3600,
    )