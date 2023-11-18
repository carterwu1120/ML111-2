import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

train_data = pd.DataFrame({
    # rich:1 -> rich
    # rich:0 -> poor
    'rich':[1, 0, 0, 0, 1, 1, 0, 1, 1, 0],
    'GDP':[6, 0.3, 1, 0.8, 5, 4, 1, 3.3, 4.1, 0.65],
    'labor':[20, 5, 6, 3, 50, 35, 10, 15, 48, 20],
    'locals':[10, 40, 30, 20, 16, 8, 43, 3, 14, 60]
})
test_data = pd.DataFrame({
    'rich':[1],
    'GDP':[3],
    'labor':[15],
    'locals':[10]
})

# x1 = data['locals'].values
# x2 = data['GDP'].values
x_train = train_data[['GDP', 'labor']].values
y_train = train_data['rich'].values
x_test = test_data[['GDP', 'labor']].values

logisticModel = LogisticRegression(random_state=0)
logisticModel.fit(x_train, y_train)
predict = logisticModel.predict(x_test)


print(predict)


# # k-neraest neighbors
# neighbors = []
# for i in range(len(y)):
#   neighbors.append([np.sqrt((x1[i]-test_x1)**2 + (x2[i]-test_x2)**2), i])
  

# neighbors.sort(key=lambda x: x[0])

# print(neighbors)
# k = 5
# k_neighbors = []
# for i in range(k):
#   k_neighbors.append(y[neighbors[i][1]])
# print(k_neighbors)
# if k_neighbors.count(1) > k_neighbors.count(0):
#   print('This is a rich country')
# else:
#   print('This is not a rich country')