
import pandas as pd
import nearest_neighbour
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/wang215/Documents/GitHub/quant0/Sample_data/stk_min1/sh600000_1min_sample.csv')

# plt.plot(df.index, df['open'], 'k')
# plt.plot(df.index, df['max'], 'r')
# plt.plot(df.index, df['min'], 'b')
# plt.plot(df.index, df['close'], 'g')
# plt.show()

data = df['open']
d = 50
m = 3
k = 20
method_1 = 'correlation'
method_2 = 'absolute_distance'
[OutSample_For_Corr, InSample_For_Corr, InSample_Res_Corr] = nearest_neighbour.nn(data, d, m, k, method_1)
[OutSample_For_Abs, InSample_For_Abs, InSample_Res_Abs] = nearest_neighbour.nn(data, d, m, k, method_2)

plt.plot(df.index[d:], data[d:], 'k', df.index[d:], InSample_For_Corr, 'r', df.index[d:], InSample_For_Abs, 'b')
plt.show()
