# export DATABASE_URL="postgresql://cancerscan:xjb88xz5q2wcp632Q2KMaw@cancerscan-10499.7tt.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full"

from flask import Flask, render_template, request, redirect, url_for, flash, make_response, Blueprint, jsonify, Response
from replit import Database
from argon2 import PasswordHasher
from passlib.hash import sha256_crypt
import openai
import requests
from PIL import Image
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
import psycopg2
from passlib.hash import bcrypt  # for password hashing
import os
import cv2
import math
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Connect to the CockroachDB cluster
conn = psycopg2.connect(os.environ["DATABASE_URL"])

# Create a table to store user information
cur = conn.cursor()
cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(50) NOT NULL,
        password VARCHAR(100) NOT NULL
    );
""")
conn.commit()

def add_user(username, password):
    hashed_password = bcrypt.hash(password)
    cur = conn.cursor()
    cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
    conn.commit()

def user_exists(username):
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = %s", (username,))
    return cur.fetchone() is not None

def authenticate(username, password):
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cur.fetchone()
    if user is None:
        return False
    hashed_password = user[2]
    return bcrypt.verify(password, hashed_password)

import psycopg2

import psycopg2

def change_password(username, current_password, new_password, new_password_again):
    if new_password != new_password_again:
        return "New passwords do not match"
    
    if not authenticate(username, current_password):
        return "Incorrect username or password"
    
    hashed_password = bcrypt.hash(new_password)
    
    cur = conn.cursor()
    cur.execute("UPDATE users SET password = %s WHERE username = %s", (hashed_password, username))
    conn.commit()
    
    return "Password updated successfully"



openai.api_key = ""

ph = PasswordHasher()

db = Database(db_url="https://kv.replit.com/v0/eyJhbGciOiJIUzUxMiIsImlzcyI6ImNvbm1hbiIsImtpZCI6InByb2Q6MSIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJjb25tYW4iLCJleHAiOjE2ODI5NDI1MDEsImlhdCI6MTY4MjgzMDkwMSwiZGF0YWJhc2VfaWQiOiIyMzA1MzQxZC1kOTZlLTRjYTUtYWM1ZS1hNDhlN2ZkOGIwNGQiLCJ1c2VyIjoiQzBEM1cxWiIsInNsdWciOiJEYXRhYmFzZS1VUkwifQ.7f9BeU_alofzEVz87rwJ58PPnMzdkzMAA9rXb8g2i2NclFZzfc0cId-Sij2XwQvjHHro2qrT9vOF-W38HMhHkA")

app = Flask(__name__)


@app.route("/dashboard", methods=['POST', 'GET'])
def scan():

  loggedIn = request.cookies.get("loggedIn")
  username = request.cookies.get("username")
  session = request.cookies.get("session")
  perms = 'none'
  
  return render_template("dashboard.html", loggedIn=loggedIn)

@app.route("/mental-training", methods=['POST', 'GET'])
def mentaltraining():

  global userforlungpoints
  userforlungpoints = request.cookies.get("username")

  loggedIn = request.cookies.get("loggedIn")
  username = request.cookies.get("username")
  session = request.cookies.get("session")

  return render_template("mental-training.html", loggedIn=loggedIn)

@app.route("/physical-training", methods=['POST', 'GET'])
def physicaltraining():

  global userforlungpoints
  userforlungpoints = request.cookies.get("username")

  loggedIn = request.cookies.get("loggedIn")
  username = request.cookies.get("username")
  session = request.cookies.get("session")

  return render_template("physical-training.html", loggedIn=loggedIn)

@app.route("/rewards", methods=['POST', 'GET'])
def rewards():
  global userforlungpoints
  userforlungpoints = request.cookies.get("username")
  username = request.cookies.get("username")
  points = db[username+"points"]
  loggedIn = request.cookies.get("loggedIn")
  url = "https://api.verbwire.com/v1/nft/mint/quickMintFromFile"
  

  if request.method == "POST" and "address1" in request.form:
    address = request.form["address1"]

    if points >= 10:
        
        points -= 10

        files = {"filePath": open("static/assets/img/NFT2.png", "rb")}
        payload = {
          "allowPlatformToOperateToken": "true",
          "chain": "goerli",
          "name": "Bow of Strength",
          "description": "An NFT Standing with Cancer Fighters",
          "recipientAddress": address
        }
        headers = {
          "accept": "application/json",
          "X-API-Key": "sk_live_7b1e935c-020d-4221-98f3-f68e70f8f1ed"
        }

        response = requests.post(url, data=payload, files=files, headers=headers)

        print(response.text)

  if request.method == "POST" and "address2" in request.form:
    address = request.form["address2"]

    if points >= 20:
        
        points -= 20

        files = {"filePath": open("static/assets/img/NFT4.png", "rb")}
        payload = {
          "allowPlatformToOperateToken": "true",
          "chain": "goerli",
          "name": "Ribbon Resilience",
          "description": "An NFT Standing with Cancer Fighters",
          "recipientAddress": address
        }
        headers = {
          "accept": "application/json",
          "X-API-Key": "sk_live_7b1e935c-020d-4221-98f3-f68e70f8f1ed"
        }

        response = requests.post(url, data=payload, files=files, headers=headers)

        print(response.text)

    if request.method == "POST" and "address3" in request.form:
      if points >= 30:
        
        points -= 30
        address = request.form["address3"]

        files = {"filePath": open("static/assets/img/NFT1.png", "rb")}
        payload = {
          "allowPlatformToOperateToken": "true",
          "chain": "goerli",
          "name": "Brave Bows",
          "description": "An NFT Standing with Cancer Fighters",
          "recipientAddress": address
        }
        headers = {
          "accept": "application/json",
          "X-API-Key": "sk_live_7b1e935c-020d-4221-98f3-f68e70f8f1ed"
        }

        response = requests.post(url, data=payload, files=files, headers=headers)

        print(response.text)

  if request.method == "POST" and "address4" in request.form:
    address = request.form["address4"]

    if points >= 40:
        
        points -= 40
        files = {"filePath": open("static/assets/img/NFT3.png", "rb")}
        payload = {
          "allowPlatformToOperateToken": "true",
          "chain": "goerli",
          "name": "Healing Horizons",
          "description": "An NFT Standing with Cancer Fighters",
          "recipientAddress": address
        }
        headers = {
          "accept": "application/json",
          "X-API-Key": "sk_live_7b1e935c-020d-4221-98f3-f68e70f8f1ed"
        }

        response = requests.post(url, data=payload, files=files, headers=headers)

        print(response.text)
  
  return render_template("rewards.html", loggedIn=loggedIn, points=points)

def capture_frame():
    success, frame = camera.read()  # read the camera frame
    if success: 
        # Save the frame as an image
        cv2.imwrite('captured_frame.jpg', frame)
        print("Captured")

        image = Image.open('captured_frame.jpg')

        # Resize the image to 600x450 pixels
        resized_image = image.resize((600, 450))

        # Save the resized image
        resized_image.save('captured_frame.jpg')

def gen_frames():
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
            
def chatbot(request):
    image_url = "/assets/img/chatbot.jpeg"
    context = {"image_url": image_url}
    return render_template("lungcancer.html", context=context, request=request)

@app.route("/skin-results", methods=['POST','GET'])
def skinresults():
  capture_frame()

  image_path = './captured_frame.jpg'
  loggedIn = request.cookies.get("loggedIn")

  model_path = "./SkinCancerDetector/model/skincancermodel"
  predictor = MultiModalPredictor.load(model_path)
  username = request.cookies.get("username")

  db[username+"points"] += 5

  if __name__ == '__main__':
      predictions = predictor.predict({'image': [image_path]})
      proba = predictor.predict_proba({'image': [image_path]})
      prob = int(proba[0][predictions[0]]*100 - 30)
  return render_template("skin-results.html", prob=prob, loggedIn=loggedIn)

@app.route("/skin-cancer", methods=['POST', 'GET'])
def skincancer():

  loggedIn = request.cookies.get("loggedIn")
  username = request.cookies.get("username")
  session = request.cookies.get("session")
  
  return render_template("skincancer.html", loggedIn=loggedIn)

camera = cv2.VideoCapture(0)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

row = []

def add_row_to_csv(file_path, row_data):
    df = pd.DataFrame([row_data])
    df.to_csv(file_path, mode='a', header=False, index=False)

def contains_only_numbers(input_string):
    if input_string.isdigit():
        return True
    return False

def append_if_only_numbers(input_data):
    if contains_only_numbers(input_data):
        row.append(input_data)

@app.route("/lung-cancer", methods=['POST', 'GET'])
def lungcancer():
    global row
    global userforlungpoints
    userforlungpoints = request.cookies.get("username")
    data = request.data
    data = data.decode('utf-8')

    if data == "male":
       row.append("M")
    elif data == "female":
       row.append("F")

    if data == "yes":
       row.append("2")
    elif data == "no":
       row.append("1")

    append_if_only_numbers(data)

    print(row)

    loggedIn = request.cookies.get("loggedIn")

    if len(row) == 15:
      csv_file_path = 'LungCancerDetector/data.csv'
      add_row_to_csv(csv_file_path, row)

      df = pd.read_csv("LungCancerDetector/data.csv")
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

      model = tf.keras.models.load_model("LungCancerDetector/model/lungcancermodel100epoch")

      global var
      y_pred = model.predict(x_test)
      # y_pred_output = np.around(y_pred)
      # print(np.concatenate((y_pred_output.reshape(-1,1),y_test.values.reshape(-1,1)),1))
      var = int((y_pred[-1][-1]*100)-30)
      # print(confusion_matrix(y_test,y_pred_output))
      row = []

    return render_template("lungcancer.html", loggedIn=loggedIn)

@app.route("/addpoints", methods=['GET','POST'])
def points():
  pointadd = request.data
  pointadd = int(pointadd.decode('utf-8'))
  db[userforlungpoints+"points"] += pointadd
  return "Worked"

@app.route("/getvar", methods=["GET","POST"])
def getvar():
  return jsonify({'variable': var})

@app.route("/gettext", methods=["GET","POST"])
def gettext():
  text = '60'
  return jsonify({'text': text})

@app.route("/chatbot", methods=['POST', 'GET'])
def chatbot():

  loggedIn = request.cookies.get("loggedIn")
  username = request.cookies.get("username")
  session = request.cookies.get("session")
  
  return render_template("chatbot.html", loggedIn=loggedIn)

@app.route("/")
def welcome():
  loggedIn = request.cookies.get("loggedIn")
  username = request.cookies.get("username")
  session = request.cookies.get("session")

  return render_template("index.html", loggedIn=loggedIn)


@app.route("/settings", methods=["GET", "POST"])
def settings():

  loggedIn = request.cookies.get("loggedIn")
  username = request.cookies.get("username")
  session = request.cookies.get("session")
  
  if request.method == "POST":    
    currpass = request.form.get("currpass")
    newpass = request.form.get("newpass")
    repass = request.form.get("repass")

    change_password(username, currpass, newpass, repass)

  if loggedIn == "true":
    if username != None and user_exists(username) == True:
      if ph.verify(session, username) == True:
        return render_template("settings.html", loggedIn = loggedIn, username = username, session = session)
    else:
      return redirect("/logout")
  else:
    return render_template("notloggedin.html", loggedIn = loggedIn, username = username, session = session)

@app.route("/login")
def login():
  loggedIn = request.cookies.get("loggedIn")
  username = request.cookies.get("username")
  session = request.cookies.get("session")
  perms = 'none'
  if loggedIn == "true":
    if username != None and user_exists(username) == True:
      if ph.verify(session, username) == True:
        perms = db[username+"stat"]
  if loggedIn == "true":
    return redirect("/")
  else:
    return render_template("login.html", loggedIn = loggedIn, username = username, session = session)

@app.route("/signup")
def signup():
  loggedIn = request.cookies.get("loggedIn")
  username = request.cookies.get("username")
  session = request.cookies.get("session")
  perms = 'none'
  if loggedIn == "true":
    if username != None and user_exists(username) == True:
      if ph.verify(session, username) == True:
        perms = db[username+"stat"]
  if loggedIn == "true":
    return redirect("/")
  else:
    return render_template("signup.html", loggedIn = loggedIn, username = username, session = session)

@app.route("/loginsubmit", methods=["GET", "POST"])
def loginsubmit():
  if request.method == "POST":
    username = request.form.get("username")
    password = request.form.get("password")
    if user_exists(username):
      if authenticate(username, password) == True:
        resp = make_response(render_template('readcookie.html'))
        resp.set_cookie("loggedIn", "true")
        resp.set_cookie("username", username)
        resp.set_cookie("session", ph.hash(username))
        return resp
      else:
        return render_template("error.html", error="Incorrect Password, please try again.")
    else:
      return render_template("error.html", error="Account does not exist, please sign up.")

@app.route("/createaccount", methods=["GET", "POST"])
def createaccount():
  if request.method == "POST":
    newusername = request.form.get("newusername")
    newpassword = sha256_crypt.encrypt((request.form.get("newpassword")))
    orignewpass = request.form.get("newpassword")
    reenterpassword = request.form.get("reenterpassword")
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    cap_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    allchars = letters + cap_letters + numbers + ['_']
    for i in newusername:
      if i not in allchars:
        return "Username can only contain alphanumeric characters and underscores."
    if user_exists(newusername) == True:
      return render_template("error.html", error="Username taken.")
    if newusername == "":
      return render_template("error.html", error="Please enter a username.")
    if newpassword == "":
      return render_template("error.html", error="Please enter a password.")
    if reenterpassword == orignewpass:
      add_user(newusername, orignewpass)
      db[newusername+"points"] = 0
      resp = make_response(render_template('readcookie.html'))
      resp.set_cookie("loggedIn", "true")
      resp.set_cookie("username", newusername)
      resp.set_cookie("session", ph.hash(newusername))
      return resp
    else:
      return render_template("error.html", error="Passwords don't match.")

@app.route("/logout")
def logout():
  resp = make_response(render_template('readcookie.html'))
  resp.set_cookie("loggedIn", "false")
  resp.set_cookie("username", "None")
  return resp

if __name__ == "__main__":
  app.run(debug=True, port=5000, host='0.0.0.0')