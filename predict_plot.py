import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

###时序图
data=pd.read_csv('./data/ETTh1.csv').iloc[:,1:]
print(data.shape)
col_names = data.columns
pred=np.load('./results/predict_24.npy')
true=np.load('./results/true_24.npy')

#所有节点预测_真实值在一张图上
fig = plt.figure(figsize=(24, 24))
for i in range(len(col_names)):
  ax = fig.add_subplot(len(col_names),1,i+1)
  ax.plot(true[:250,i],label='Ground Truth')
  ax.plot(pred[:250,i],label='Predict')
  ax.set_title(col_names[i])
  ax.set_xlabel('Time Step')
  ax.set_ylabel('Value')
  plt.legend()
fig.tight_layout(pad=3.0)
plt.savefig('./figures/ETTh1_allnodes_predict.png')
plt.show()
plt.close()

#节点预测_真实值分别绘制并保存
for i in range(len(col_names)):
    plt.plot(true[:250,i],label='Ground Truth')
    plt.plot(pred[:250,i],label='Predict')
    plt.title(col_names[i])
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend(loc='upper right')
    plt.savefig('./figures/ETTh1_node{}_predict.png'.format(i))
    plt.close()
