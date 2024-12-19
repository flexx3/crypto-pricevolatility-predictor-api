#Libraries to prepare data
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import sqlite3
from app.data import api_data, SqlRepository
load_dotenv()
#import garch model library
from arch import arch_model
#libraries to save and load model
import joblib
from glob import glob
from pathlib import Path

class Garch_model:
    def __init__(self, ticker):
        self.ticker = ticker
        #instantiate name for the filepath for volatility model sub directory
        self.model_directory = os.environ.get('Model_directory')
        self.model2_subdirectory = os.environ.get('model2_subdirectory')
        
    #wrangle function to get data
    def wrangle(self, ticker, use_new_data=True):
        #set up connection to db
        connection = sqlite3.connect(database= os.environ.get('DB_NAME'), check_same_thread= False)
        #instantiate sql repo
        repo = SqlRepository(connection= connection)
        cursor = connection.cursor()
        if use_new_data is True:
            #instantiate api_data
            api = api_data()
            records = api.get_data(ticker)
            #format the columns
            if records.columns.to_list() != ['Close', 'High', 'Low', 'Open', 'Volume']:
                column_list= records.columns.to_list()
                columns= [val[0] for val in column_list]
                records.columns= columns
            query = f"Drop Table If Exists '{ticker}' "
            cursor.execute(query)
            connection.commit()
            data = repo.insert_data(table_name= ticker, records= records, if_exists= 'replace')
        df = repo.read_table(ticker)
        connection.close()
        #calculate returns
        df['Returns'] = df['Close'].pct_change()*100
        df.sort_values(by='Date', inplace=True)
        df.fillna(method='ffill', inplace=True)        
        return df['Returns'].dropna()
        
    #create and fit model
    def fit(self, p=1, q=1):
        data= self.wrangle(self.ticker, use_new_data=True)
        self.model = arch_model(data, p=p, q=q, mean='Zero', vol='Garch').fit(disp='off')
    #format predictions
    def __format_predictions(self, forecasts):
        #get start date
        start = forecasts.index[0] + pd.DateOffset(days=1)
        #get index for the predictions series
        date_range = pd.bdate_range(start=start, periods=forecasts.shape[1])
        #calculate the predicted future daily volatility
        values = forecasts.values.flatten() **0.5
        #pack them up into a  pandas series 
        series = pd.Series(data=values, index=date_range)
        return series  
        
    #create forecast for volatility
    def forecast_volatility(self, horizon):
        forecasts = self.model.forecast(horizon=horizon, reindex=False).variance
        formatted_forecast = self.__format_predictions(forecasts)
        return formatted_forecast
    
   #save model
    def dump(self):
       #create file path to save and store the volatility model
        filepath = os.path.join(self.model_directory, self.model2_subdirectory,(f"{self.ticker}.pkl"))
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        #save model
        joblib.dump(self.model, filepath)
        return filepath
    #load model
    def load(self):
        #prepare a pattern for glob search
        pattern = os.path.join(self.model_directory, self.model2_subdirectory, (f"*{self.ticker}.pkl"))
        try:
            model_path = sorted(glob(pattern))[-1]
        except IndexError:
            raise Exception(f"Oops No model trained for {self.ticker} chai..")
        self.model = joblib.load(model_path)
        return self.model
    
