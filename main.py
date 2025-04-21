import streamlit as st

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from statsmodels import api
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as lreg
from sklearn.metrics import mean_squared_log_error as msle, r2_score as r2s, mean_squared_error as mse, mean_absolute_error as mae

import os
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', None, 'display.max_columns', None)
plt.style.use('dark_background')
st.set_page_config("CCU Scoring System", layout='wide')

st.title("Carbon Capture Unit Scoring System")

os.system("cls")

def plotCorr():
    plt.figure(figsize=(20, 7))
    sn.heatmap(df.corr(), annot=True, cmap="GnBu")
    plt.show()

df = pd.read_csv("ccu_synthetic_data.csv")

scores = {
    0: "Ineffective",
    1: "Very Poor",
    2: "Poor",
    3: "Average",
    4: "Above Average",
    5: "Excellent"
}

x = df[[col for col in df.columns if col not in ['Effectiveness_Score', 'CCU_ID']]]
y = df['Effectiveness_Score']

xtrain, xtest, ytrain, ytest = tts(x, y, random_state=20, test_size=0.2)
lr = lreg()
lr.fit(xtrain, ytrain)

c1, c2, c3 = st.columns(3)
ccuquantity = c1.number_input("How many CCUs do you want to evaluate?", min_value=0, max_value=10, value=1, step=1)

st.write("---")

xpred = pd.DataFrame()

c1, c2, c3 = st.columns(3)

xpred['Cost_per_ton_USD'] = []
xpred['Efficiency_percent'] = []
xpred['Capacity_tons_per_year'] = []
xpred['Energy_consumption_MWh_per_ton'] = []
xpred['Age_years'] = []
xpred['Ambient_temperature_C'] = []

for i in range(ccuquantity):

    st.header(f"CCU {i+1}:")

    c1, c2, c3 = st.columns(3)

    concatdf = pd.DataFrame()

    concatdf['Cost_per_ton_USD'] = [c2.number_input(f"Cost Per Metric Ton of CCU {i+1} (USD):", value=40.00, step=0.01)]
    concatdf['Efficiency_percent'] = [c1.number_input(f"CCU {i+1} Efficiency (%):", value=85.0, step=0.1)]
    concatdf['Capacity_tons_per_year'] = [c3.number_input(f"CCU {i+1} Capture Capacity (Metric Tons/Year):", value=1000000, step=1)]
    concatdf['Energy_consumption_MWh_per_ton'] = [c1.number_input(f"CCU {i+1} Energy Consumption Rate (MWh/Metric Ton):", value=3.00, step=0.01)]
    concatdf['Age_years'] = [c3.number_input(f"CCU {i+1} Age (Years):", value=2.5, step=0.1)]
    concatdf['Ambient_temperature_C'] = [c2.number_input(f"CCU {i+1} Ambient Capture Site Temperature (°C):", value=25.0, step=0.01)]

    xpred = pd.concat([xpred, concatdf])

    st.write("---")

st.expander("Input Values Preview").dataframe(xpred, hide_index=False, use_container_width=True)

if st.button("Predict"):
    
    pred = lr.predict(xpred)
    predict = {}

    for i in range(len(pred)):
        pred[i] = round(pred[i])
        predict[i] = int(pred[i])

    if len(predict) == 1:
        st.write(f"**CCU Score:** `{predict[0]}` - ***{scores[predict[0]]}***")

    else:

        st.subheader("CCU Scores:")

        for ccu in predict:
            st.write(f"**CCU {ccu+1}:** {predict[ccu]} - ***{scores[predict[ccu]]}***")