import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, linregress

# a sample dataset with advertising and sales
data = pd.DataFrame({
    'SocialMedia': [340.1, 154.5, 127.2, 261.5, 290.8, 115.7, 167.5, 230.2, 115.6, 309.8, 176.1, 324.7, 130.8, 207.5, 314.1, 305.4, 177.8, 391.4, 179.2, 257.3],
    'TV': [147.8, 150.3, 157.9, 151.3, 133.8, 156.9, 140.8, 127.6, 109.1, 109.6, 112.8, 131.0, 142.1, 130.6, 140.9, 170.7, 149.6, 153.6, 130.5, 130.9],
    'Billboard': [169.2, 145.1, 169.3, 157.5, 157.4, 182.0, 130.5, 118.6, 108.0, 128.2, 123.2, 103.0, 164.9, 106.2, 145.0, 151.9, 213.0, 154.8, 117.3, 126.1],
    'Sample':[96.2, 97.5, 98.1, 98.7, 99.3, 100.1, 100.9, 101.2, 102.5, 103.8, 104.3, 105.1, 105.8, 106.4, 107.2, 107.8, 108.4, 108.9, 109.2, 109.7],
    'Sales': [29.1, 17.4, 16.3, 25.5, 19.9, 14.2, 18.8, 20.2, 11.8, 18.6, 16.6, 23.4, 15.2, 15.7, 26.0, 29.4, 19.5, 31.4, 18.3, 21.6]
})

# reshape the Numpy array into a two-dimensional array with a single column
x = data['SocialMedia'].values
y = data['Sales'].values

# Slope and intercept of the regression line
numerator = np.sum((x - np.mean(x)) * (y - np.mean(y)))
denominator = np.sum((x - np.mean(x)) ** 2)

slope = numerator / denominator
intercept = np.mean(y) - slope * np.mean(x)

# predict sales based on advertising spend
y_pred = intercept + slope * x

# calculate the RSS
RSS = np.sum((y - y_pred) ** 2)
n = len(data['SocialMedia'])

var = np.sqrt(RSS/(n-2))
SE = var/np.sqrt(np.sum((x-np.mean(x))**2))

# T-statistic
t_statistic =  slope / SE
print("t-statistic: ", t_statistic)

df = n-2
# One-tailed p-value
if t_statistic < 0:
    p_val = t.cdf(t_statistic, df=df) # t.cdf: cumulative distribution function (CDF) of the Student's t-distribution
else:
    p_val = 1 - t.cdf(t_statistic, df=df)
print("one-tailed p-value: ", p_val)

if(p_val < 0.05):
    print('Reject the null hypothesis') 
else:
    print('Accept the null hypothesis') 

# Two-tailed p-value
if t_statistic < 0:
    p_val = t.cdf(t_statistic, df=df) * 2
else:
    p_val = (1 - t.cdf(t_statistic, df=df)) * 2
print("two-tailed p-value: ", p_val)

if(p_val < 0.05):
    print('Reject the null hypothesis') 
else:
    print('Accept the null hypothesis') 

# plot the data and regression line
plt.scatter(x, y, color='darkblue')
plt.plot(x, y_pred, color='red', alpha=0.5)

# print the slope, intercept, and RSS
plt.text(x.min()+0.25*(x.max()-x.min()), y.max()-0.25*(y.max()-y.min()), f"Slope: {slope:.2f}\nIntercept: {intercept:.2f}\nRSS: {RSS:.2f}", fontsize=8)

# set the title and axis labels
plt.title('Advertising (SocialMedia)')
plt.xlabel('Advertising Spend (1000$)')
plt.ylabel('Sales')

# show the plot
plt.show()

# reshape the Numpy array into a two-dimensional array with a single column
x1 = data['TV'].values
y = data['Sales'].values
#  Slope and intercept of the regression line
numerator = np.sum((x1 - np.mean(x1)) * (y - np.mean(y)))
print(numerator)
denominator = np.sum((x1 - np.mean(x1)) ** 2)

slope = numerator / denominator
intercept = np.mean(y) - slope * np.mean(x1)

# predict sales based on advertising spend
y1_pred = intercept + slope * x1

# calculate the RSS
RSS = np.sum((y - y_pred) ** 2)
n = len(data['TV'])

var = np.sqrt(RSS/(n-2))
SE = var/np.sqrt(np.sum((x1-np.mean(x1))**2))

# T-statistic
t_statistic =  slope / SE
print("t-statistic: ", t_statistic)

df = n-2
# One-tailed p-value
if t_statistic < 0:
    p_val = t.cdf(t_statistic, df=df) # t.cdf: cumulative distribution function (CDF) of the Student's t-distribution
else:
    p_val = 1 - t.cdf(t_statistic, df=df)
print("one-tailed p-value: ", p_val)

if(p_val < 0.05):
    print('Reject the null hypothesis') 
else:
    print('Accept the null hypothesis') 

# Two-tailed p-value
if t_statistic < 0:
    p_val = t.cdf(t_statistic, df=df) * 2
else:
    p_val = (1 - t.cdf(t_statistic, df=df)) * 2
print("two-tailed p-value: ", p_val)

if(p_val < 0.05):
    print('Reject the null hypothesis') 
else:
    print('Accept the null hypothesis') 

# plot the data and regression line
plt.scatter(x1, y, color='darkblue')
plt.plot(x1, y1_pred, color='red', alpha=0.5)

# print the slope, intercept, and RSS
plt.text(x1.min()+0.25*(x1.max()-x1.min()), y.max()-0.25*(y.max()-y.min()), f"Slope: {slope:.2f}\nIntercept: {intercept:.2f}\nRSS: {RSS:.2f}", fontsize=8)

# set the title and axis labels
plt.title('Advertising (TV)')
plt.xlabel('Advertising Spend (1000$)')
plt.ylabel('Sales')

# show the plot
plt.show()

