### ---------------------------------------------------------------------------------------------------------- ###
### ----------------------------------------- TAIL HEDGE PORTFOLIO ------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### PACKAGES ###
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
### ----------------------------------------- TAIL HEDGE PORTFOLIO ------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def plot_tsunami_model(start, end, **kwargs):
    with open(Path(DATA_DIR) / 'VIX.csv', 'rb') as file:
        vix = pd.read_csv(file)
    with open(Path(DATA_DIR) / 'VVIX.csv', 'rb') as file:
        vvix = pd.read_csv(file)

    vix.index = pd.to_datetime(vix['Date']).values
    vix = pd.DataFrame(vix['Close'])
    vix.columns = ['vix']
    vvix.index = pd.to_datetime(vvix['Date']).values
    vvix = pd.DataFrame(vvix['Close'])
    vvix.columns = ['vvix']

def plot_veqtor_vix(start, end, **kwargs):
    with open(Path(DATA_DIR) / 'SPX.csv', 'rb') as file:
        spx = pd.read_csv(file)
    spx.index = pd.to_datetime(spx['Date']).values
    factor_df = pd.DataFrame(spx['Close'])
    factor_df.columns = ['spx']








