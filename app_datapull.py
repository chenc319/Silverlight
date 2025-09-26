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

    ### SP500 SECTORS ###
    url = 'https://en.macromicro.me/api/etf/us/timeseries/intro_main/XLE'
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Referer": "https://en.macromicro.me/etf/us/xle",
        "Origin": "https://en.macromicro.me",
        "Accept-Language": "en-US,en;q=0.9",
        "X-Requested-With": "XMLHttpRequest"
    }

    # Example: If your browser cookie is "sessionid=abc123; csrftoken=xyz789"
    cookie_string = "aiExplainOn=off; _ga=GA1.1.1326173032.1758173367; __lt__cid=5f8f1041-fc84-4aa9-abbc-4d7e15562472; _fbp=fb.1.1758173367120.77837946912536112; PHPSESSID=suobqmb0n6kkhtp8g3mvltitfv; __lt__sid=9afde53c-4ab7fa56; cf_clearance=sQcnw1W.jIT3uhFu25g8ZUhs8vzvBmGC0PhoIrK3KUk-1758861299-1.2.1.1-SXm45k7qHH0VDqz4XukLAMFna9KCofyYWCb1dTcyhvfCN..99z3xXXQVBC8EBJFtiGTbieeDDMg9dXcZ0iCcAvDzEVlkI1loTWrBuQXUiekzhrrkRuwu3zgAmuHbK1y1MGnM1UijoyyobAIvq.o.N5uNwj_KoQqeAb1k_126ya_6elfbemyiZV5R3AC2vBp4zqwE_sZiGKKYzJkj373IDZ2pMXLpFmEhNUd_5HDYMME; 3daysleft=1758861385; app_ui_support_btn=5; mm_sess_pages=4; _ga_4CS94JJY2M=GS2.1.s1758861297$o2$g1$t1758861614$j60$l0$h0"

    # Convert cookie string to dictionary
    cookies = dict([cookie.strip().split('=') for cookie in cookie_string.split(';')])

    response = requests.get(url, headers=headers, cookies=cookies)
    print(response.text)
