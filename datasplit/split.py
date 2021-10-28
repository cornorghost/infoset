import pandas as pd
import os
from sklearn.model_selection import StratifiedShuffleSplit

base_path='./'
data = pd.read_csv(os.path.join(base_path, 'data.csv'), sep='\t', header=None)

print(data.iloc[:,0])
print(data.iloc[:,1])

strat_train_set, strat_test_set = [], []
# 然后使用分层采样
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data.iloc[:,0], data.iloc[:,1]):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

print(strat_train_set[1])