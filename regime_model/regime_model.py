### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------- REGIME MODEL ---------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### PACKAGES ###
import functools as ft
import requests
from pandas_datareader import data as pdr
from pathlib import Path
import os
import pickle
from io import StringIO
import pandas as pd
import requests
import base64
import pandas as pd
from io import StringIO
DATA_DIR = os.getenv('DATA_DIR', 'data')

def merge_dfs(array_of_dfs):
    return ft.reduce(lambda left,
                            right: pd.merge(left,
                                            right,
                                            left_index=True,
                                            right_index=True,
                                            how='outer'),
                     array_of_dfs)

### DATA PULL ###
start = '1990-01-01'
end = pd.to_datetime('today')


iorb = pdr.DataReader('GDP', 'fred', start, end)
with open(Path(DATA_DIR) / 'iorb.pkl', 'wb') as file:
    pickle.dump(iorb, file)
fed_funds = pdr.DataReader('EFFR', 'fred', start, end)



