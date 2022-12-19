# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 13:42:00 2022

"""
# 1. Library imports
import uvicorn
from fastapi import FastAPI
#from BankNotes import BankNote
import pickle
import pandas as pd
from zipfile import ZipFile

# 1. Create the app object
app = FastAPI()
pickle_in = open("LGBMClassifier.pkl","rb")
classifier=pickle.load(pickle_in)

# Ouverture des fichiers
z = ZipFile("X_data.zip")
df = pd.read_csv(z.open('X_data.csv'),
                     index_col='SK_ID_CURR', encoding ='utf-8')
#df = pd.read_csv("../data/X_test.csv")
df = df.drop('TARGET', axis=1)

from pydantic import BaseModel
# 2. Class which describes Bank Notes measurements
class request_id(BaseModel):
    SK_ID_CURR: int

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}


# 4. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_banknote(data: request_id):
    
    SK_ID = data.SK_ID_CURR
    
    df_pred= pd.DataFrame(classifier.predict(df), index=df.index)
    df_prob= pd.DataFrame(classifier.predict_proba(df), index=df.index)
    
    prediction = df_pred.loc[df_pred.index == SK_ID].values.tolist()
    prob2 = df_prob.loc[df_prob.index == SK_ID].values.tolist()
    
    return {'Customer_ID': f'{SK_ID}','Prediction': f'{prediction}', 'Default Probability %': f'{round((prob2[0][1])*100,2)}'}

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000) #, debug=True)

