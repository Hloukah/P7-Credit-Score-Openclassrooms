# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 23:16:12 2022

@author: Allihamdulilahi
"""
from flask import Flask, request #, render_template
import pandas as pd
#import numpy as np
#import gc
import pickle
#import sqlite3
import flask
from zipfile import ZipFile
#pipeline contains the entire pipeline for prediction of query point
#from flask import Flask, jsonify, render_template
#from flask import request
#import json
#import time

#from pipeline import final_pipeline

app = Flask(__name__)

pickle_in = open("LGBMClassifier.pkl","rb")
classifier=pickle.load(pickle_in)

# Ouverture des fichiers
z = ZipFile("X_data.zip")
df = pd.read_csv(z.open('X_data.csv'),
                     index_col='SK_ID_CURR', encoding ='utf-8')
#df = pd.read_csv("../data/X_test.csv")
df = df.drop('TARGET', axis=1)

#z = ZipFile("default_risk.zip")
#data = pd.read_csv(z.open('default_risk.csv'),
#                    index_col='SK_ID_CURR', encoding ='utf-8')


#home page
@app.route('/', methods = ['GET'])
def home_page():
    return flask.render_template('home-page.html')

      
    #prediction page
    @app.route('/home', methods = ['POST', 'GET'])
    def inputs_page():
    	return flask.render_template('predict.html')

    #results page
    @app.route('/predict', methods = ['POST'])
    def prediction():
    	#getting the SK_ID_CURR from user
    	#sk_id_curr = request.form.to_dict()['SK_ID_CURR']
        SK_ID = request.form.get("SK_ID_CURR")
        SK_ID = int(SK_ID)
        
        #session["name"] = request.form.get("name")
        df_pred= pd.DataFrame(classifier.predict(df), index=df.index)
        df_prob= pd.DataFrame(classifier.predict_proba(df), index=df.index)
        
        predictin = df_pred.loc[df_pred.index == SK_ID].values.tolist()
        prob2 = df_prob.loc[df_prob.index == SK_ID].values.tolist()
        
        if predictin == 1:
       		prediction = 'a Potential Defaulter'
        else:
        	prediction = 'not a Defaulter'
        predicted_proba = round((prob2[0][1])*100,2)
        #df_income = pd.DataFrame(["AMT_INCOME_TOTAL"])

        #data_for_display = pd.concat([test_datapoint[['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE']], data_for_display.reset_index(drop = True)], axis = 1)
        #data_for_display = data_for_display.to_html(classes = 'data', header = 'true', index = False)

    	#conn.close()
    	#gc.collect()

        return flask.render_template('result_and_inference.html',#   tables = [data_for_display],
    		output_proba = predicted_proba, output_class = prediction, sk_id_curr = SK_ID)

if __name__ == '__main__':
	app.run(host = '0.0.0.0', port = 5000)
