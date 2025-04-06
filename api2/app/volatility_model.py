#Libraries to prepare data
import os
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import polars as pl
import duckdb
from sqlalchemy import create_engine, text
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
    def wrangle(self, ticker):
        #get current date str
        today= datetime.now().strftime('%Y-%m-%d')
        #setup conection to db
        engine= create_engine(f"duckdb:///{os.environ.get('DB_NAME')}")
        with engine.connect() as conn:
            #check if table exists
            result= conn.execute(text(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{self.ticker}'"))
            table_name= result.fetchone()
        #if table name exists    
        if table_name is not None:
            #instantiate 'sqlrepo' class from data.py library
            repo= SqlRepository(uri=f"duckdb:///{os.environ.get('DB_NAME')}")
            #load data from table
            df= repo.read_table(self.ticker)
            #check if data corresponds to the most recent date
            if (df.is_empty()) or (df['Date'].max() != datetime.strptime(today, '%Y-%m-%d').date()):
                api= api_data(self.ticker)
                data= api.get_data()
                repo.insert_data(table_name=self.ticker, records=data)
                #load data from table
                df= repo.read_table(self.ticker)
            else:
                df= df
        #if table does not exists in database
        else:
            #instantiate 'api_data' class from data.py library
            api= api_data(self.ticker)
            data= api.get_data()
            #instantiate 'sqlrepo' class from data.py library
            repo= SqlRepository(uri=f"duckdb:///{os.environ.get('DB_NAME')}")
            #setup connection to execute and commit changes to the db based on the below query
            with engine.connect() as conn:
                conn.execute(text(f'Drop Table If Exists "{self.ticker}"'))
                conn.commit()
            #insert data into database
            repo.insert_data(table_name=self.ticker, records=data)
            #load data from table
            df=repo.read_table(self.ticker)
        return df
        
    #create and fit model
    def fit(self, p=1, q=1):
        data= self.wrangle(self.ticker)
        data= data.to_pandas()
        data.set_index('Date', inplace=True)
        data.sort_values(by='Date', inplace=True)
        data['Return']= data['Close'].pct_change()*100
        data.dropna(axis=0, inplace=True)
        self.model = arch_model(data['Return'], p=p, q=q, mean='Zero', vol='Garch').fit(disp='off')
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
    
