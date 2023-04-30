import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

df = pd.read_csv("./data.csv")
df.head()

maper = {"M":1, "F":0, 1:0, 2:1, "YES":1, "NO":0}
df["AGE_Catagory"] = pd.cut(df["AGE"],bins=[0,20,40,65,120],labels=[0,1,2,3])
df_2 = df.drop(["AGE","AGE_Catagory"],axis=1)
df_2 = df_2.applymap(lambda x: maper.get(x))
df_2["AGE_catagory"] = df["AGE_Catagory"]
df_2

X = df_2.drop("LUNG_CANCER",axis=1)
y = df_2["LUNG_CANCER"]
ct = ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[-1])],remainder="passthrough")
X = ct.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

model = tf.keras.models.load_model("./model/lungcancermodel100epoch")

y_pred = model.predict(x_test)
# y_pred_output = np.around(y_pred)
# print(np.concatenate((y_pred_output.reshape(-1,1),y_test.values.reshape(-1,1)),1))
print(y_pred)
# print(confusion_matrix(y_test,y_pred_output))