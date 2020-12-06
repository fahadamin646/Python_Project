from flask import Flask
app = Flask(__name__)
import pandas as pd
import pickle
import Process as pr
@app.route('/')
def hello_world():
    pred = pr.getPrediction(0,0,0,0,1)
    return pred