import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv("train.csv")
df = pd.DataFrame(data)

x = np.array(df['mse']).reshape(-1,1)
y = np.array(df['ese']).reshape(-1,1)

rm = LinearRegression()
rm.fit(x,y)

prediction = rm.predict([[1.50]])
print('Expected battery %f' % prediction)
