#import api dependencies
from fastapi import FastAPI
from pydantic import BaseModel
#import model dependencies
import os
import requests
from dotenv import load_dotenv
from app.volatility_model import Garch_model

#instantiate the fastapi app
app= FastAPI()

@app.get('/hello', status_code=200)
def show_hello():
    return{'message': 'path is smooth'}

#declare fitIn,fitOut class
class fitIn(BaseModel):
    ticker: str
class fitOut(fitIn):
    success: bool
    message: str
        
#create post method for '/fit' path with 200 status-code
@app.post('/fit', status_code= 200, response_model= fitOut)
def fit_model(request:fitIn):
    #create response dict
    response= request.dict()
    #create try block
    try:
        #instantiate model
        model_class= Garch_model(request.ticker)
        #fit model
        model_class.fit()
        #save model
        filename= model_class.dump()
        #add success key to response
        response["success"]= True
        #add message key to response
        response["message"]= f"volatility model saved and trained for {filename}."
    #create except block
    except Exception as e:
        response["success"]= False
        response["message"]= str(e)
    return response

#declare predictOut, predictIn class
class predictIn(BaseModel):
    ticker: str
    horizon: int
class predictOut(predictIn):
    success: bool
    message: str
    forecasts: dict 
#create post for /predict path
@app.post('/predict', status_code=200, response_model= predictOut)
def make_forecast(request:predictIn):
    #create response dict
    response= request.dict()
    #create try block
    try:
        #instantiate model
        model= Garch_model(request.ticker)
        #load saved model
        model.load()
        #make forecasts
        forecasts= model.forecast_volatility(request.horizon)
        #add success key to response
        response['success']= True
        #add message key to response
        response['message']= f'Projected volatility for {request.horizon} days.'
        #add forecasts key to response
        response['forecasts']= forecasts.to_dict()
    #create except block
    except Exception as e:
        #add success key to response
        response['success']= False
        #add message key to response
        response['message']= str(e)
        #add forecasts key to response
        response['forecasts']= {}
    return response
        