
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#important features that we calculated with random forest classifier
feature_names = ['L3_S38_F3960', 'L3_S33_F3865', 'L3_S38_F3956', 'L3_S33_F3857',
       'L3_S29_F3321', 'L1_S24_F1846', 'L3_S32_F3850', 'L3_S29_F3354',
       'L3_S29_F3324', 'L3_S35_F3889', 'L0_S1_F28', 'L1_S24_F1844',
       'L3_S29_F3376', 'L0_S0_F22', 'L3_S33_F3859', 'L3_S38_F3952', 
       'L3_S30_F3754', 'L2_S26_F3113', 'L3_S30_F3759', 'L0_S5_F114']

#numeric data import
numeric_cols = pd.read_csv('C:/Users/Admin/Desktop/bosch-production-line-performance/train_numeric.csv.zip', nrows = 1).columns.values
imp_idxs = [np.argwhere(feature_name == numeric_cols)[0][0] for feature_name in feature_names]
train = pd.read_csv('C:/Users/Admin/Desktop/bosch-production-line-performance/train_numeric.csv.zip', 
                index_col = 0, header = 0, usecols = [0, len(numeric_cols) - 1] + imp_idxs)
train = train[feature_names + ['Response']]
X_neg, X_pos = train[train['Response'] == 0].iloc[:, :-1], train[train['Response']==1].iloc[:, :-1] #dividing data based on failed and not failed features ( parts that fail quality control (represented by a 'Response' = 1))


FIGSIZE=(12,16)
non_missing = pd.DataFrame(pd.concat([(X_neg.count()/X_neg.shape[0]).to_frame('uspjeli uzorci'),
                                      (X_pos.count()/X_pos.shape[0]).to_frame('neuspjeli uzorci'),  
                                      ], 
                       axis = 1))
non_missing_sort = non_missing.sort_values(['uspjeli uzorci'])
non_missing_sort.plot.barh(title = 'Udio vrijednosti koje ne nedostaju', figsize = FIGSIZE) #how many non missing values in overall data (again data is divied based on failed and not falied products)

plt.gca().invert_yaxis()
plt.show()


FIGSIZE = (13,4)
_, (ax1, ax2) = plt.subplots(1,2, figsize = FIGSIZE)
MIN_PERIODS = 100

triang_mask = np.zeros((X_pos.shape[1], X_pos.shape[1]))
triang_mask[np.triu_indices_from(triang_mask)] = True
#matrix of correlation for parts that failed
ax1.set_title('neuspjela klasa')
sns.heatmap(X_neg.corr(min_periods = MIN_PERIODS), mask = triang_mask, square=True,  ax = ax1)
#matrix of correlation for parts that did not failed
ax2.set_title('uspjela klasa')
sns.heatmap(X_pos.corr(min_periods = MIN_PERIODS), mask = triang_mask, square=True,  ax = ax2)
plt.show()
