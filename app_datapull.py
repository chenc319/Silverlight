### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------- DATA PULLS ------------------------------------------------ ###
### ---------------------------------------------------------------------------------------------------------- ###

import pandas as pd
import functools as ft
import streamlit as st
import plotly.graph_objs as go
from pandas_datareader import data as pdr
from matplotlib import pyplot as plt
from pathlib import Path
import os
import pickle
from plotly.subplots import make_subplots
import numpy as np
DATA_DIR = os.getenv('DATA_DIR', 'data')

def merge_dfs(array_of_dfs):
    return ft.reduce(lambda left, right: pd.merge(left, right,
                                                  left_index=True,
                                                  right_index=True, how='outer'), array_of_dfs)

### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------- DATA PULLS ------------------------------------------------ ###
### ---------------------------------------------------------------------------------------------------------- ###

start = '1900-01-01'
end = pd.to_datetime('today')
def refresh_data(start,end,**kwargs):
    growth = pdr.DataReader('USALOLITOAASTSAM',
                            'fred',
                            start,
                            end).resample('ME').last()
    with open(Path(DATA_DIR) / 'growth.pkl', 'wb') as file:
        pickle.dump(growth, file)
    inflation = pdr.DataReader('CPIAUCSL',
                               'fred',
                               start,
                               end).resample('ME').last()
    with open(Path(DATA_DIR) / 'inflation.pkl', 'wb') as file:
        pickle.dump(inflation, file)

    treasury_1m = pdr.DataReader('DGS1MO', 'fred', start, end)
    with open(Path(DATA_DIR) / 'treasury_1m.pkl', 'wb') as file:
        pickle.dump(treasury_1m, file)

    treasury_3m = pdr.DataReader('DGS3MO', 'fred', start, end)
    with open(Path(DATA_DIR) / 'treasury_3m.pkl', 'wb') as file:
        pickle.dump(treasury_3m, file)

    treasury_6m = pdr.DataReader('DGS6MO', 'fred', start, end)
    with open(Path(DATA_DIR) / 'treasury_6m.pkl', 'wb') as file:
        pickle.dump(treasury_6m, file)

    treasury_1y = pdr.DataReader('DGS1', 'fred', start, end)
    with open(Path(DATA_DIR) / 'treasury_1y.pkl', 'wb') as file:
        pickle.dump(treasury_1y, file)

    treasury_2y = pdr.DataReader('DGS2', 'fred', start, end)
    with open(Path(DATA_DIR) / 'treasury_2y.pkl', 'wb') as file:
        pickle.dump(treasury_2y, file)

    treasury_3y = pdr.DataReader('DGS3', 'fred', start, end)
    with open(Path(DATA_DIR) / 'treasury_3y.pkl', 'wb') as file:
        pickle.dump(treasury_3y, file)

    treasury_5y = pdr.DataReader('DGS5', 'fred', start, end)
    with open(Path(DATA_DIR) / 'treasury_5y.pkl', 'wb') as file:
        pickle.dump(treasury_5y, file)

    treasury_7y = pdr.DataReader('DGS7', 'fred', start, end)
    with open(Path(DATA_DIR) / 'treasury_7y.pkl', 'wb') as file:
        pickle.dump(treasury_7y, file)

    treasury_10y = pdr.DataReader('DGS10', 'fred', start, end)
    with open(Path(DATA_DIR) / 'treasury_10y.pkl', 'wb') as file:
        pickle.dump(treasury_10y, file)

    treasury_20y = pdr.DataReader('DGS20', 'fred', start, end)
    with open(Path(DATA_DIR) / 'treasury_20y.pkl', 'wb') as file:
        pickle.dump(treasury_20y, file)

    treasury_30y = pdr.DataReader('DGS30', 'fred', start, end)
    with open(Path(DATA_DIR) / 'treasury_30y.pkl', 'wb') as file:
        pickle.dump(treasury_30y, file)


