import re 
import os
from flask import Flask,app, render_template, request
import numpy as np
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.python.ops.gen_array_ops import concat

model=load_model(r"coal1.h5")

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/prediction.html')
def prediction():
	return render_template('prediction.html')

@app.route('/index.html')
def home():
	return render_template('index.html')

@app.route('/result', methods=["GET","POST"])
def res():
   if request.method=="POST":
    f=request.files['image']
    basepath=os.path.dirname(__file__)
    filepath=os.path.join(basepath,'static',f.filename)
    f.save(filepath)
    img=image.load_img(filepath,target_size=(128,128))
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    index=["Anthracite","Bituminous","Lignite","Peat"]
    prediction=model.predict(x)
    result=str(index[prediction[0].tolist().index(1)])
    return render_template('prediction.html',prediction=result)


 
if __name__ == "__main__":
    app.run(debug = True)
