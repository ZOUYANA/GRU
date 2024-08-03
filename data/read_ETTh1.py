import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_csv = pd.read_csv('ETTh1.csv')
data_mtx = data_csv.iloc[:, 1:].values

n_steps = data_mtx.shape[0]
n_nodes = data_mtx.shape[1]

# plot
for i in range(n_nodes):
    plt.subplot(3, 3, i + 1)
    plt.plot(data_mtx[:, i])
plt.show()
