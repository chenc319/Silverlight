import pandas as pd
from pathlib import Path
import os
import pickle
import requests
import re
from datetime import datetime

DATA_DIR = os.getenv('DATA_DIR', 'data')

# # Define the range from the first report date to today
# start_date = "2010-07-20"
# end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
# tuesdays = pd.date_range(start=start_date, end=end_date, freq='W-TUE')
# wednesdays = pd.date_range(start=start_date, end=end_date, freq='W-WED')
# all_report_dates = sorted(list(set(tuesdays) | set(wednesdays)))
# all_report_dates_str = [d.strftime("%Y-%m-%d") for d in all_report_dates]
#
# historical_cftf_financial_texts = {}
# for report_date in all_report_dates_str:
#     date_obj = datetime.strptime(report_date, "%Y-%m-%d")
#     numeric_date = date_obj.strftime("%m%d%y")
#     year_str = date_obj.strftime("%Y")
#     url = ('https://www.cftc.gov/sites/default/files/files/dea/cotarchives/'
#            + year_str + '/futures/financial_lf' + numeric_date + '.htm')
#     try:
#         response = requests.get(url)
#         if response.status_code == 404:
#             continue  # Skip this iteration if the page is not found
#         text = response.text
#         historical_cftf_financial_texts[report_date] = text
#         print(report_date)
#     except requests.RequestException:
#         continue  # Skip on other request errors too
#
# with open(Path(DATA_DIR) / 'historical_cftf_financial_texts.pkl', 'wb') as file:
#     pickle.dump(historical_cftf_financial_texts, file)


with open(Path(DATA_DIR) / 'historical_cftf_financial_texts.pkl', 'rb') as file:
    historical_cftf_financial_texts = pd.read_pickle(file)

def parse_block(block):
    lines = block.splitlines()
    # Robust open interest search (search whole block just in case)
    oi_match = re.search(r"Open Interest is\s*([\d,]+)", block)
    open_interest = int(oi_match.group(1).replace(",", "")) if oi_match else None

    # Find contract name as before
    open_interest_idx = None
    for i, line in enumerate(lines):
        if "Open Interest is" in line:
            open_interest_idx = i
            break
    contract = "Unknown"
    if open_interest_idx is not None and open_interest_idx > 0:
        contract_line = lines[open_interest_idx - 1].strip()
        contract = re.sub(r"\(.*", "", contract_line).strip()

    pos_line_idx = [i for i, l in enumerate(lines) if l.strip().startswith("Positions")]
    data_line_idx = pos_line_idx[0] + 1 if pos_line_idx else -1
    if data_line_idx != -1:
        data_line = lines[data_line_idx]
        data_fields = re.findall(r'\d[\d,]*', data_line)
        columns = [
            "dealer_long", "dealer_short", "dealer_spread",
            "asset_mgr_long", "asset_mgr_short", "asset_mgr_spread",
            "lev_funds_long", "lev_funds_short", "lev_funds_spread",
            "other_long", "other_short", "other_spread",
            "nonrep_long", "nonrep_short"
        ]
        values = [int(f.replace(",", "")) for f in data_fields[:len(columns)]]
        values = values + [None] * (len(columns) - len(values))
        row = dict(contract=contract, open_interest=open_interest, **dict(zip(columns, values)))
        return row
    return None

historical_cftf_financial_dfs = {}
for each_date, each_text in historical_cftf_financial_texts.items():
    blocks = re.split(r"Traders in Financial Futures - Futures Only Positions as of [A-Za-z]+\s+\d+, \d{4}\s*-+\s*",
                      each_text)
    blocks = [block for block in blocks if "Positions" in block and "Open Interest" in block]
    records = [parse_block(b) for b in blocks]
    records = [r for r in records if r]
    historical_cftf_financial_dfs[each_date] = pd.DataFrame(records)
    print(each_date)

with open(Path(DATA_DIR) / 'historical_cftf_financial_dfs.pkl', 'wb') as file:
    pickle.dump(historical_cftf_financial_dfs, file)

spx_positioning_df = pd.DataFrame(columns = [
    'open_interest',
    'dealer_long', 'dealer_short', 'dealer_spread',
    'asset_mgr_long', 'asset_mgr_short', 'asset_mgr_spread',
    'lev_funds_long', 'lev_funds_short', 'lev_funds_spread'
],
    index = list(historical_cftf_financial_dfs.keys())
)

emini_spx_positioning_df = pd.DataFrame(columns = [
    'open_interest',
    'dealer_long', 'dealer_short', 'dealer_spread',
    'asset_mgr_long', 'asset_mgr_short', 'asset_mgr_spread',
    'lev_funds_long', 'lev_funds_short', 'lev_funds_spread'
],
    index = list(historical_cftf_financial_dfs.keys())
)

vix_positioning_df = pd.DataFrame(columns = [
    'open_interest',
    'dealer_long', 'dealer_short', 'dealer_spread',
    'asset_mgr_long', 'asset_mgr_short', 'asset_mgr_spread',
    'lev_funds_long', 'lev_funds_short', 'lev_funds_spread'
],
    index = list(historical_cftf_financial_dfs.keys())
)

for each_date, each_text in historical_cftf_financial_dfs.items():
    try:
        df = each_text.copy()
        df.index = each_text['contract']
        spx_positioning_df.loc[each_date,spx_positioning_df.columns] = (
            df.loc)['S&P 500 Consolidated - CHICAGO MERCANTILE EXCHANGE',spx_positioning_df.columns]
        emini_spx_positioning_df.loc[each_date,emini_spx_positioning_df.columns] = (
            df.loc)['E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE',emini_spx_positioning_df.columns]
        vix_positioning_df.loc[each_date,vix_positioning_df.columns] = (
            df.loc)['VIX FUTURES - CBOE FUTURES EXCHANGE',vix_positioning_df.columns]
    except:
        continue

spx_positioning_df = spx_positioning_df.dropna()
emini_spx_positioning_df = emini_spx_positioning_df.dropna()
vix_positioning_df = vix_positioning_df.dropna()

with open(Path(DATA_DIR) / 'spx_positioning_df.pkl', 'wb') as file:
    pickle.dump(spx_positioning_df, file)
with open(Path(DATA_DIR) / 'emini_spx_positioning_df.pkl', 'wb') as file:
    pickle.dump(emini_spx_positioning_df, file)
with open(Path(DATA_DIR) / 'vix_positioning_df.pkl', 'wb') as file:
    pickle.dump(vix_positioning_df, file)

