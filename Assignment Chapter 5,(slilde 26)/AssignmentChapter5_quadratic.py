import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor

# advertising and sales
data = pd.DataFrame({
    'SocialMedia': [370.4, 154.5, 127.2, 241.5, 240.8, 115.7, 157.5, 230.2, 134.3, 309.48, 146.4, 324.7, 120.8, 207.5, 311.1, 325.5, 277.6, 331.4, 159.4, 157.2],
    'Sales': [25.1, 17.3, 12.6, 25.5, 19.9, 14.8, 18.8, 20.2, 11.7, 18.6, 16.6, 23.9, 15.2, 15.7, 26.2, 29.24, 19.4, 31.55, 18.23, 23.6]
})
# reshape the Numpy array into a two-dimensional array with a single column
x = data['SocialMedia'].values.reshape(-1, 1)
y = data['Sales'].values

# Number of data samples
n = len(y)
val_size = n//10
train_size = n-val_size
K = 10
learning_rate = 0.00000000001  # *For quadratic*
x_sequence = np.linspace(x.min(), x.max(), len(x))
iteration = 100
sum_MSE = 0


for k in range(K):
    val_x = x[k*val_size : (k+1)*val_size]
    val_y = y[k*val_size : (k+1)*val_size]

    train_x = np.concatenate([x[:k*val_size], x[(k+1)*val_size:]], axis=0)
    train_y = np.concatenate([y[:k*val_size], y[(k+1)*val_size:]], axis=0)
    # predict sales based on advertising spend
    theta0 = 1
    theta1 = 0.1
    theta2 = 0.1

    y_pred = theta0 + theta1 * train_x + theta2 * (train_x**2) # *For quadratic*

    RSS = np.sum((train_y - y_pred) ** 2)
    mse = np.mean((train_y - y_pred) ** 2)
    mini_batch = train_size//2


    for i in range(iteration): # learning iterations
        ii = mini_batch % (i+1)
        mini_batch_x = train_x[ii : ii+mini_batch]
        mini_batch_y = train_y[ii : ii+mini_batch]
        # Calculate the gradients and partial derivatives
        d_theta0 = (1 / mini_batch) * np.sum(y_pred - mini_batch_y)
        d_theta1 = (1 / mini_batch) * np.sum((y_pred - mini_batch_y) * mini_batch_x.reshape(-1))
        d_theta2 = (1 / mini_batch) * np.sum((y_pred - mini_batch_y) * mini_batch_x.reshape(-1)**2) # *For quadratic*

        # plot the data 
        plt.scatter(x, y, color='darkblue')
        plt.scatter(mini_batch_x, mini_batch_y, color='red')

        # plot the  regression line
        plt.plot(x_sequence, theta0 + theta1 * x_sequence + theta2 * x_sequence**2, color='red', alpha=0.5) # *For quadratic*

        #--------------Update of the prediction-------------
        theta0 = theta0 - learning_rate * d_theta0
        theta1 = theta1 - learning_rate * d_theta1
        theta2 = theta2 - learning_rate * d_theta2 # *For quadratic*

        # Predict again to show new Plot
        y_pred = theta0 + theta1 * mini_batch_x + theta2 * (mini_batch_x**2) # *For quadratic*

        # calculate the new RSS and MSE
        New_RSS = np.sum((mini_batch_y - y_pred) ** 2)
        New_mse = np.mean((mini_batch_y - y_pred) ** 2)

        # plot new regression line 
        plt.plot(x_sequence, theta0 + theta1 * x_sequence + theta2 * x_sequence**2, color='green', alpha=0.5) # *For quadratic*

        # Plot text in fixed position
        ax = plt.gca()
        ax.annotate(f"K: {k:.0f}", xy=(0.1, 1.1), fontsize=10, color='black', xycoords='axes fraction')
        ax.annotate(f"Iteration: {i:.0f}", xy=(0.4, 1.1), fontsize=10, color='blue', xycoords='axes fraction')
        ax.annotate(f"MSE New: {New_mse:.8f}", xy=(0.05, 0.70), fontsize=10, color='green', xycoords='axes fraction')
        ax.annotate(f"MSE: {mse:.2f}", xy=(0.05, 0.75), fontsize=10, color='red', xycoords='axes fraction')
        # ax.annotate(f"new teta_0: {theta0:.6f}", xy=(0.05, 0.80), fontsize=10, xycoords='axes fraction')
        # ax.annotate(f"new teta_1: {theta1:.6f}", xy=(0.05, 0.85), fontsize=10, xycoords='axes fraction')
        # ax.annotate(f"Partial derivative teta_0: {d_theta0:.2f}", xy=(0.05, 0.90), fontsize=10, xycoords='axes fraction')
        # ax.annotate(f"Partial derivative teta_1: {d_theta1:.2f}", xy=(0.05, 0.95), fontsize=10, xycoords='axes fraction')
        diff_MSE = abs(mse - New_mse)
        RSS = New_RSS
        mse = New_mse

        # show the plot
        plt.draw()
        plt.pause(0.0001)

    
        if diff_MSE<=0.001:
            # plt.show() # To stop at end plot
            break
        else:
            plt.clf()  # Iterate Plot

    y_pred = theta0 + theta1 * val_x + theta2 * (val_x**2)
    RSS = np.sum((y_pred - val_y)**2)
    MSE = RSS/val_size
    sum_MSE += MSE
    print("K:", k, "  MSE: ", MSE, "  iter: ", i)
    

total_MSE = sum_MSE/K
print("loss: ",total_MSE)

# Note that you can use library also (e.g.): 
# from sklearn.linear_model import SGDRegressor
# model = SGDRegressor(loss='squared_error', penalty=None, learning_rate='constant', eta0=0.00000000001, max_iter=200, random_state=10)
# model.fit(x_poly, y)


# # split data into train data and test data
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=val_size)

# # biuld regression model
# model = SGDRegressor(loss='squared_error', penalty=None, learning_rate='constant', eta0=0.0000001, max_iter=200, random_state=10)
# model.fit(x_train, y_train)

# print('R-squared:', model.score(x_train, y_train))

# y_pred = model.predict(x_test)
# mse = mean_squared_error(y_test, y_pred)    # RSS = np.sum((y_pred - y_test)**2), print("MSE: ", RSS/val_size)
# print("MSE: ", mse)


# cv_score = cross_val_score(model, x, y, cv=K)
# print("cv score mean: ", cv_score.mean())
# print(cv_score)

# y_pred = model.predict(x_test)
# mse = mean_squared_error(y_test, y_pred)   
# print("MSE: ", mse)