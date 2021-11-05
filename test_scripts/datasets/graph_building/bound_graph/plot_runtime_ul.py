import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('sizes.csv', index_col=0)

for i in range(df.shape[0]):
    plt.plot(df.columns, df.iloc[i,:], label = f'Num Hops = {df.index[i]}')

#plt.title('Runtime Analysis of 1-2 Bounded Subgraph Generation')
plt.title('Sizes of 1-2 Bounded Subgraph Generation')
plt.xlabel('Number of Initial Subgraphs')
plt.ylabel('Time (sec)')
plt.legend()
plt.show()