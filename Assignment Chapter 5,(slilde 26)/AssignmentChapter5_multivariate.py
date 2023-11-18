import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error


# advertising and sales
# a sample dataset with advertising and sales
data = pd.DataFrame({
    'SocialMedia': [340.1, 154.5, 127.2, 261.5, 290.8, 115.7, 167.5, 230.2, 115.6, 309.8, 176.1, 324.7, 130.8, 207.5, 314.1, 305.4, 177.8, 391.4, 179.2, 257.3],
    'TV': [147.8, 150.3, 157.9, 151.3, 133.8, 156.9, 140.8, 127.6, 109.1, 109.6, 112.8, 131.0, 142.1, 130.6, 140.9, 170.7, 149.6, 153.6, 130.5, 130.9],
    'Billboard': [169.2, 145.1, 169.3, 157.5, 157.4, 182.0, 130.5, 118.6, 108.0, 128.2, 123.2, 103.0, 164.9, 106.2, 145.0, 151.9, 213.0, 154.8, 117.3, 126.1],
    'Sample':[96.2, 97.5, 98.1, 98.7, 99.3, 100.1, 100.9, 101.2, 102.5, 103.8, 104.3, 105.1, 105.8, 106.4, 107.2, 107.8, 108.4, 108.9, 109.2, 109.7],
    'Sales': [29.1, 17.4, 16.3, 25.5, 19.9, 14.2, 18.8, 20.2, 11.8, 18.6, 16.6, 23.4, 15.2, 15.7, 26.0, 29.4, 19.5, 31.4, 18.3, 21.6]
})
# reshape the Numpy array into a two-dimensional array with a single column
x = data[['SocialMedia','TV', 'Billboard']].values
y = data['Sales'].values

K = 10
sum_MSE = 0

model = SGDRegressor(loss='squared_error', penalty=None, learning_rate='constant', eta0=0.00001, max_iter=200, random_state=10)

kf = KFold(n_splits=10)
for k, (train_index, test_index) in  enumerate(kf.split(x)):
    # print(f"Fold {i}:")
    # print(f"  Train: index={train_index}")
    # print(f"  Test:  index={test_index}")
    val_x = x[test_index]
    val_y = y[test_index]
    train_x = x[train_index]
    train_y = y[train_index]
    
    model.fit(train_x, train_y)
    
    y_pred = model.predict(val_x)

    MSE = mean_squared_error(val_y, y_pred)

    print("K:", k, "  MSE: ", MSE)
    sum_MSE += MSE
total_MSE = sum_MSE/K
print("loss: ", total_MSE)
