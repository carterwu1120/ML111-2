import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


# advertising and sales
data = pd.DataFrame({
    'SocialMedia': [370.4, 154.5, 127.2, 241.5, 240.8, 115.7, 157.5, 230.2, 134.3, 309.48, 146.4, 324.7, 120.8, 207.5, 311.1, 325.5, 277.6, 331.4, 159.4, 157.2],
    'Sales': [25.1, 17.3, 12.6, 25.5, 19.9, 14.8, 18.8, 20.2, 11.7, 18.6, 16.6, 23.9, 15.2, 15.7, 26.2, 29.24, 19.4, 31.55, 18.23, 23.6]
})
# reshape the Numpy array into a two-dimensional array with a single column
x = data['SocialMedia'].values.reshape(-1, 1)
y = data['Sales'].values

# predict sales based on advertising spend
theta0 = 1
theta1 = 0.1
theta2 = 0.1

y_pred = theta0 + theta1 * x + theta2 * (x**2) # *For quadratic*

# calculate the RSS and MSE
RSS = np.sum((y - y_pred) ** 2)
mse = np.mean((y - y_pred) ** 2)

# learning loop iteration
iteration = 1000

# Update the model parameters using gradient descent
learning_rate = 0.000000000001  # *For quadratic*

# For ploting create a sequence of x
x_sequence = np.linspace(x.min(), x.max(), len(x))

# set the title and axis labels
plt.title('Advertising (SocialMedia)')
plt.xlabel('Advertising Spend (1000$)')
plt.ylabel('Sales')
plt.ion()
plt.ioff()

for i in range(iteration): # learning iterations

    # Calculate the gradients and partial derivatives
    n = len(y)  # Number of data samples
    d_theta0 = (1 / n) * np.sum(y_pred - y)
    d_theta1 = (1 / n) * np.sum((y_pred - y) * x.reshape(-1))
    d_theta2 = (1 / n) * np.sum((y_pred - y) * x.reshape(-1)**2) # *For quadratic*

    # plot the data 
    plt.scatter(x, y, color='darkblue')

    # plot the  regression line
    plt.plot(x_sequence, theta0 + theta1 * x_sequence + theta2 * x_sequence**2, color='red', alpha=0.5) # *For quadratic*

    #--------------Update of the prediction-------------
    theta0 = theta0 - learning_rate * d_theta0
    theta1 = theta1 - learning_rate * d_theta1
    theta2 = theta2 - learning_rate * d_theta2 # *For quadratic*

    # Predict again to show new Plot
    y_pred = theta0 + theta1 * x + theta2 * (x**2) # *For quadratic*

    # calculate the new RSS and MSE
    New_RSS = np.sum((y - y_pred) ** 2)
    New_mse = np.mean((y - y_pred) ** 2)

    # plot new regression line 
    plt.plot(x_sequence, theta0 + theta1 * x_sequence + theta2 * x_sequence**2, color='green', alpha=0.5) # *For quadratic*

    # Plot text in fixed position
    ax = plt.gca()
    ax.annotate(f"Iteration: {i:.0f}", xy=(0.4, 1.1), fontsize=10, color='blue', xycoords='axes fraction')
    ax.annotate(f"MSE New: {New_mse:.8f}", xy=(0.05, 0.70), fontsize=10, color='green', xycoords='axes fraction')
    ax.annotate(f"MSE: {mse:.2f}", xy=(0.05, 0.75), fontsize=10, color='red', xycoords='axes fraction')
    # ax.annotate(f"new teta_0: {theta0:.6f}", xy=(0.05, 0.80), fontsize=10, xycoords='axes fraction')
    # ax.annotate(f"new teta_1: {theta1:.6f}", xy=(0.05, 0.85), fontsize=10, xycoords='axes fraction')
    # ax.annotate(f"Partial derivative teta_0: {d_theta0:.2f}", xy=(0.05, 0.90), fontsize=10, xycoords='axes fraction')
    # ax.annotate(f"Partial derivative teta_1: {d_theta1:.2f}", xy=(0.05, 0.95), fontsize=10, xycoords='axes fraction')

    RSS = New_RSS
    mse = New_mse

    # show the plot
    plt.draw()
    plt.pause(0.0001)

   
    if i == iteration - 1:
        plt.show() # To stop at end plot
    else:
        plt.clf()  # Iterate Plot
    
# Note that you can use library also (e.g.): 
# from sklearn.linear_model import SGDRegressor
# model = SGDRegressor(loss='squared_error', penalty=None, learning_rate='constant', eta0=0.00000000001, max_iter=200, random_state=10)
# model.fit(x_poly, y)