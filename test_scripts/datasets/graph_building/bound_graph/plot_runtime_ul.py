import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_csv('runtime_sizes/12-12/sizes.csv', index_col=0)

# for i in range(df.shape[0]):
#     plt.plot(df.columns, np.log(df.iloc[i,:]), label = f'Num Hops = {df.index[i]}')

# #plt.title('Runtime Analysis of 1-2 Bounded Subgraph Generation')
# plt.title('Sizes of 1-2 Bounded Subgraph Generation')
# plt.xlabel('Number of Initial Subgraphs')
# plt.ylabel('log(Time (sec))')
# plt.legend()
# plt.show()

df_sizes = pd.read_csv('runtime_sizes/12-12/sizes.csv', index_col=0)
df_runtimes = pd.read_csv('runtime_sizes/12-12/runtimes.csv', index_col = 0)

fig, (ax1, ax2) = plt.subplots(1, 2)

for i in range(df_sizes.shape[0]):
    ax1.plot(df_sizes.columns, df_sizes.iloc[i,:], label = f'p = {df_sizes.index[i]}')
    ax2.plot(df_runtimes.columns, np.log(df_runtimes.iloc[i,:]), label = f'p = {df_runtimes.index[i]}')

#plt.title('Runtime Analysis of 1-2 Bounded Subgraph Generation')
ax1.set_title('Sizes of 1-2 ShapeGraph Generation')
ax1.set_xlabel('Number of Initial Subgraphs')
ax1.set_ylabel('Num. Nodes')
ax1.legend()

ax2.set_title('Runtime of ShapeGraph Generation')
ax2.set_xlabel('Number of Initial Subgraphs')
ax2.set_ylabel('log(Time (sec))')
ax2.legend()

plt.show()