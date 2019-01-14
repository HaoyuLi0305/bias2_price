# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 13:41:58 2018

@author: 李浩宇
"""

import pandas as pd
import numpy as np
df=pd.read_csv("E://ceci.csv")
ts = df['综合价'].dropna()


testStationarity(ts)
plt.plot(ts)
draw_acf_pacf(ts)

#调参
from statsmodels.tsa.arima_model import ARMA
for i in range(9):
    for j in range(2):
        model=ARMA(ts, order=(i, j))
        result_arma = model.fit( disp=-1)
        aic=result_arma.aic
        print("i=%s , j=%s ,aic=%s"%(i,j,aic))
        
model = ARMA(ts, order=(2, 1))
result_arma = model.fit( disp=-1)
plt.plot(ts)
plt.plot(result_arma.fittedvalues,color='red')
#
predict = result_arma.predict(47, 53, dynamic=True)