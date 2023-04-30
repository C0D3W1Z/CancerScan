import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("./input/lungcancer.csv")
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

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
r = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=300)

print(f"train score : {model.evaluate(x_train,y_train)[1]}")
print(f"test score : {model.evaluate(x_test,y_test)[1]}")

model.save("./model/lungcancermodel300epoch")