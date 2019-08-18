import pandas as pd
import numpy as np

marks = pd.read_csv("gradData.csv")
df = pd.DataFrame(marks)

mse_mark = np.array(df['mse']).reshape(-1,1)
ese_mark = np.array(df['ese']).reshape(-1,1)

m = 0
c = 0
lr = 0.0001
epoch = 1000

n = float(len(mse_mark))
for i in range(epoch):
    ypred = (mse_mark*m) + c
    dm = (-2/n)*sum(mse_mark * (ese_mark - ypred))
    de = (-2/n)*sum(ese_mark - ypred)
    m = m - (lr*dm)
    c = c - (lr*de)
    print(m,c)
    
