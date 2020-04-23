import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

X1, Y1 = make_blobs(n_samples=[400, 600, 400], n_features=2, random_state=1)

mms = MinMaxScaler()
mms.fit(X1)
X1 = mms.transform(X1)

df = pd.DataFrame(X1, columns=['x1', 'x2'])
df.insert(2, 'class', Y1)
df.to_csv('data/blob_data.csv', index=False)

print(df)

plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, s=25, edgecolor='k')
plt.show()



