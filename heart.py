# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 13:14:26 2020

@author: VAISHNAVI GOPALUNI
"""


from flask import Flask
from flask import request, render_template
from keras.models import load_model
import tensorflow as tf
model=load_model('heart.h5')
import numpy as np
global graph
graph = tf.get_default_graph()
app = Flask(__name__)
@app.route('/')
def hello_world():
    return render_template("heart.html")
@app.route('/predict', methods=["POST"])
def hello_world1():
    a = request.form["age1"]
    b=request.form["op1"]
    c=request.form["bps"]
    d=request.form["ch"]
    e=request.form["tha"]
    f=request.form["old"]
    total=[[a,b,c,d,e,f]]

    label=["NO","YES"]
    with graph.as_default():
        pred=model.predict(np.array(total))
        if(pred>0.5):
            return render_template("heart.html",y="YES")
        else:
            return render_template("heart.html",y="NO")
    
if __name__ == '__main__':
    app.run(debug=True)