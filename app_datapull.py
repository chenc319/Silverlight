### ---------------------------------------------------------------------------------------------------------- ###
### ---------------------------------------------- DATA PULLS ------------------------------------------------ ###
### ---------------------------------------------------------------------------------------------------------- ###

import pandas as pd
import functools as ft
from pandas_datareader import data as pdr
from pathlib import Path
import os
import pickle
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
                            end).resample('ME').last().shift(1)
    with open(Path(DATA_DIR) / 'growth.pkl', 'wb') as file:
        pickle.dump(growth, file)

    ### GROWTH VARIABLES ###
    initial_claims = pdr.DataReader('ICSA','fred',start,end).resample('ME').last().shift(1)
    with open(Path(DATA_DIR) / 'initial_claims.pkl', 'wb') as file:
        pickle.dump(initial_claims, file)
    di_reserves = pdr.DataReader('TOTRESNS','fred',start,end).resample('ME').last().shift(1)
    with open(Path(DATA_DIR) / 'di_reserves.pkl', 'wb') as file:
        pickle.dump(di_reserves, file)
    currency_in_circulation = pdr.DataReader('CURRCIR', 'fred', start, end).resample('ME').last().shift(1)
    with open(Path(DATA_DIR) / 'currency_in_circulation.pkl', 'wb') as file:
        pickle.dump(currency_in_circulation, file)
    m2_money_supply = pdr.DataReader('M2SL', 'fred', start, end).resample('ME').last().shift(1)
    with open(Path(DATA_DIR) / 'm2_money_supply.pkl', 'wb') as file:
        pickle.dump(m2_money_supply, file)
    real_pce = pdr.DataReader('PCEC96', 'fred', start, end).resample('ME').last().shift(1)
    with open(Path(DATA_DIR) / 'real_pce.pkl', 'wb') as file:
        pickle.dump(real_pce, file)
    retail_sales = pdr.DataReader('RETAILSMSA','fred',start,end).resample('ME').last().shift(1)
    with open(Path(DATA_DIR) / 'retail_sales.pkl', 'wb') as file:
        pickle.dump(retail_sales, file)
    advanced_retail_sales = pdr.DataReader('RSXFS','fred',start,end).resample('ME').last().shift(1)
    with open(Path(DATA_DIR) / 'advanced_retail_sales.pkl', 'wb') as file:
        pickle.dump(advanced_retail_sales, file)
    ind_prod_total_index = pdr.DataReader('INDPRO','fred',start,end).resample('ME').last().shift(1)
    with open(Path(DATA_DIR) / 'ind_prod_total_index.pkl', 'wb') as file:
        pickle.dump(ind_prod_total_index, file)
    indu_prod_manufacturing_idx = pdr.DataReader('IPMAN','fred',start,end).resample('ME').last().shift(1)
    with open(Path(DATA_DIR) / 'indu_prod_manufacturing_idx.pkl', 'wb') as file:
        pickle.dump(indu_prod_manufacturing_idx, file)
    ind_prod_cons_goods = pdr.DataReader('IPCONGD','fred',start,end).resample('ME').last().shift(1)
    with open(Path(DATA_DIR) / 'ind_prod_cons_goods.pkl', 'wb') as file:
        pickle.dump(ind_prod_cons_goods, file)
    total_nonfarm_payrolls = pdr.DataReader('PAYEMS','fred',start,end).resample('ME').last().shift(1)
    with open(Path(DATA_DIR) / 'total_nonfarm_payrolls.pkl', 'wb') as file:
        pickle.dump(total_nonfarm_payrolls, file)
    unemployment_rate = pdr.DataReader('UNRATE','fred',start,end).resample('ME').last().shift(1)
    with open(Path(DATA_DIR) / 'unemployment_rate.pkl', 'wb') as file:
        pickle.dump(unemployment_rate, file)
    retail_trade_employment = pdr.DataReader('CES0600000007','fred',start,end).resample('ME').last().shift(1)
    with open(Path(DATA_DIR) / 'retail_trade_employment.pkl', 'wb') as file:
        pickle.dump(retail_trade_employment, file)
    pce_total = pdr.DataReader('PCEC','fred',start,end).resample('ME').last().shift(1)
    with open(Path(DATA_DIR) / 'pce_total.pkl', 'wb') as file:
        pickle.dump(pce_total, file)
    pce_chain = pdr.DataReader('PCECA','fred',start,end).resample('ME').last().shift(1)
    with open(Path(DATA_DIR) / 'pce_chain.pkl', 'wb') as file:
        pickle.dump(pce_chain, file)
    pce_goods = pdr.DataReader('pce_goods','fred',start,end).resample('ME').last().shift(1)
    with open(Path(DATA_DIR) / 'pce_goods.pkl', 'wb') as file:
        pickle.dump(pce_goods, file)
    pce_durable_goods = pdr.DataReader('PCEDG','fred',start,end).resample('ME').last().shift(1)
    with open(Path(DATA_DIR) / 'pce_durable_goods.pkl', 'wb') as file:
        pickle.dump(pce_durable_goods, file)

    growth_variables_merge = merge_dfs([
        real_pce, growth, retail_sales, advanced_retail_sales, ind_prod_total_index,
        indu_prod_manufacturing_idx, ind_prod_cons_goods, total_nonfarm_payrolls,
        unemployment_rate, retail_trade_employment, pce_goods, pce_durable_goods
    ]).dropna()
    with open(Path(DATA_DIR) / 'growth_variables_merge.pkl', 'wb') as file:
        pickle.dump(growth_variables_merge, file)


    ### INFLATION VARIABLES ###
    cpi_total = pdr.DataReader('CPIAUCSL', 'fred', start, end).resample('ME').last().shift(1)
    cpi_less_foodenergy = pdr.DataReader('CPILFESL', 'fred', start, end).resample('ME').last().shift(1)
    ppi_total = pdr.DataReader('PPIACO', 'fred', start, end).resample('ME').last().shift(1)
    cpi_food = pdr.DataReader('CPIUFDSL', 'fred', start, end).resample('ME').last().shift(1)
    cpi_energy = pdr.DataReader('CPIENGSL', 'fred', start, end).resample('ME').last().shift(1)
    cpi_household_furnishings = pdr.DataReader('CUSR0000SAH3', 'fred', start, end).resample('ME').last().shift(1)
    cpi_apparel = pdr.DataReader('CPIAPPSL', 'fred', start, end).resample('ME').last().shift(1)
    cpi_medical_care = pdr.DataReader('CPIMEDSL', 'fred', start, end).resample('ME').last().shift(1)
    cpi_transportation = pdr.DataReader('CPITRNSL', 'fred', start, end).resample('ME').last().shift(1)
    cpi_alcohol = pdr.DataReader('CUSR0000SAF116', 'fred', start, end).resample('ME').last().shift(1)
    cpi_motor_fuel = pdr.DataReader('CUSR0000SETB', 'fred', start, end).resample('ME').last().shift(1)
    cpi_services_less_energy = pdr.DataReader('CUSR0000SASLE', 'fred', start, end).resample('ME').last().shift(1)
    inflation_variables_merge = merge_dfs(
        [cpi_total, cpi_less_foodenergy, ppi_total,
         cpi_food, cpi_energy, cpi_household_furnishings,
         cpi_apparel, cpi_medical_care, cpi_transportation,
         cpi_alcohol, cpi_motor_fuel, cpi_services_less_energy]).dropna()
    with open(Path(DATA_DIR) / 'inflation_variables_merge.pkl', 'wb') as file:
        pickle.dump(inflation_variables_merge, file)

    gdp = pdr.DataReader('GDPC1',
                            'fred',
                            start,
                            end).resample('ME').last()
    with open(Path(DATA_DIR) / 'gdp.pkl', 'wb') as file:
        pickle.dump(gdp, file)
    inflation = pdr.DataReader('CPIAUCSL',
                               'fred',
                               start,
                               end).resample('ME').last().shift(1)
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

    ### GROWTH VARIABLES ###
    cli_admplitude_adjusted = pdr.DataReader('USALOLITOAASTSAM', 'fred', start, end)
    industrial_production = pdr.DataReader('INDPRO', 'fred', start, end)
    trade_balance_payments_basis = pdr.DataReader('BOPGSTB', 'fred', start, end)
    advanced_retail_sales_retail_trade = pdr.DataReader('RSXFS', 'fred', start, end)
    manufacturing_spending = pdr.DataReader('TLMFGCONS', 'fred', start, end)
    all_employees_total_nonfarm = pdr.DataReader('PAYEMS', 'fred', start, end)
    goods_producing_employment = pdr.DataReader('USGOOD', 'fred', start, end)
    all_employees_manufacturing = pdr.DataReader('MANEMP', 'fred', start, end)
    avg_earnings_all_private_employees = pdr.DataReader('CES0500000011', 'fred', start, end)
    reaL_personal_expenditures = pdr.DataReader('PCEC96', 'fred', start, end)
    real_retail_food_services_sails = pdr.DataReader('RRSFS', 'fred', start, end)
    total_vehicle_sales = pdr.DataReader('TOTALSA', 'fred', start, end)

    grid_growth_variables = merge_dfs([cli_admplitude_adjusted,industrial_production,trade_balance_payments_basis,
                                            advanced_retail_sales_retail_trade,manufacturing_spending,all_employees_total_nonfarm,
                                            goods_producing_employment,all_employees_manufacturing,avg_earnings_all_private_employees,
                                            reaL_personal_expenditures,real_retail_food_services_sails,total_vehicle_sales]).resample('ME').last().shift(1).dropna()
    grid_growth_variables.index = pd.to_datetime(grid_growth_variables.index).values
    with open(Path(DATA_DIR) / 'grid_growth_variables.pkl', 'wb') as file:
        pickle.dump(grid_growth_variables, file)

    ### INFLATION VARIABLES ###
    cpi_total = pdr.DataReader('CPIAUCSL', 'fred', start, end)
    cpi_less_foodenergy = pdr.DataReader('CPILFESL', 'fred', start, end)
    ppi_total = pdr.DataReader('PPIACO', 'fred', start, end)
    cpi_food = pdr.DataReader('CPIUFDSL', 'fred', start, end)
    cpi_energy = pdr.DataReader('CPIENGSL', 'fred', start, end)
    cpi_household_furnishings = pdr.DataReader('CUSR0000SAH3', 'fred', start, end)
    cpi_apparel = pdr.DataReader('CPIAPPSL', 'fred', start, end)
    cpi_medical_care = pdr.DataReader('CPIMEDSL', 'fred', start, end)
    cpi_transportation = pdr.DataReader('CPITRNSL', 'fred', start, end)
    cpi_alcohol = pdr.DataReader('CUSR0000SAF116', 'fred', start, end)
    cpi_motor_fuel = pdr.DataReader('CUSR0000SETB', 'fred', start, end)
    cpi_services_less_energy = pdr.DataReader('CUSR0000SASLE', 'fred', start, end)


    grid_inflation_variables = merge_dfs(
        [cpi_total, cpi_less_foodenergy, ppi_total,
         cpi_food,cpi_energy,cpi_household_furnishings,
         cpi_apparel, cpi_medical_care, cpi_transportation,
         cpi_alcohol, cpi_motor_fuel,cpi_services_less_energy]).resample('ME').last().shift(
        1).dropna()
    grid_inflation_variables.index = pd.to_datetime(grid_inflation_variables.index).values
    with open(Path(DATA_DIR) / 'grid_inflation_variables.pkl', 'wb') as file:
        pickle.dump(grid_inflation_variables, file)











