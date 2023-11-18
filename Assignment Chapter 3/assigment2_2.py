import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f, linregress
from sklearn import linear_model
data = pd.DataFrame({
    'SocialMedia': [340.1, 154.5, 127.2, 261.5, 290.8, 115.7, 167.5, 230.2, 115.6, 309.8, 176.1, 324.7, 130.8, 207.5, 314.1, 305.4, 177.8, 391.4, 179.2, 257.3],
    'TV': [147.8, 150.3, 157.9, 151.3, 133.8, 156.9, 140.8, 127.6, 109.1, 109.6, 112.8, 131.0, 142.1, 130.6, 140.9, 170.7, 149.6, 153.6, 130.5, 130.9],
    'Billboard': [169.2, 145.1, 169.3, 157.5, 157.4, 182.0, 130.5, 118.6, 108.0, 128.2, 123.2, 103.0, 164.9, 106.2, 145.0, 151.9, 213.0, 154.8, 117.3, 126.1],
    'Sales': [29.1, 17.4, 16.3, 25.5, 19.9, 14.2, 18.8, 20.2, 11.8, 18.6, 16.6, 23.4, 15.2, 15.7, 26.0, 29.4, 19.5, 31.4, 18.3, 21.6],
    'sample' : [96.2, 97.5, 98.1, 98.7, 99.3, 100.1, 100.9, 101.2, 102.5, 103.8, 104.3, 105.1, 105.8, 106.4, 107.2, 107.8, 108.4, 108.9, 109.2, 109.7],
    'accept_null':[0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1]
})
x = data['sample'].values.reshape(-1,1)
y = data['Sales'].values

regr = linear_model.LinearRegression()
regr.fit(x, y)
y_pred = regr.predict(x)

RSS = np.sum((y - y_pred) ** 2)

n = len(y)
TSS = np.sum((y - np.mean(y)) ** 2)
f_statistic = ((TSS - RSS)/1) / (RSS / (n-2))
print("f_statistic", f_statistic)

critical_val = f.ppf(q=1-.05, dfn=1, dfd=8)
print("critical_val", critical_val)

if(critical_val < f_statistic):
    print('Reject the null hypothesis') 
else:
    print('Accept the null hypothesis') 


