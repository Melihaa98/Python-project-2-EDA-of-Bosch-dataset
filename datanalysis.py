import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#data import (numeric only)
#importovanje podataka.Zbog obilnosti podataka i nemogucnosti obrade svih podataka odjednom,importujemo samo dio podatak.(navodjenjem nrows parametra funkcije read_csv)

date = pd.read_csv('C:/Users/Admin/Desktop/bosch-production-line-performance/train_date.csv.zip', nrows=10000) #data with timestamp for each measurment of parts as they move through production line
numeric = pd.read_csv('C:/Users/Admin/Desktop/bosch-production-line-performance/train_numeric.csv.zip', nrows=10000) #numeric data-measurments of parts as they move thruogh production line
category = pd.read_csv('C:/Users/Admin/Desktop/bosch-production-line-performance/train_categorical.csv.zip', nrows=10000) #categorical data- -||-


#pregled broja podataka koji nedostaju jer oni prave probleme pri fitanju modela s obzirom da velika vecina modela ne radi sa NaN podacima
print('broj podatak koji nedostaje u datumskim podacima:',date.isnull().values.sum())
print('broj podatak koji nedostaje u numeričkim podacima:',numeric.isnull().values.sum())
print('broj podatka koji nedostaje u kategoričkim podacima:',category.isnull().values.sum())

#broj featura u svakom setu podatak:
print('Broj karakteristika u numeričkim podacima:',len(numeric.columns))
print('Broj karakteristika u kategoričkim podacima:',len(category.columns))
print('Broj karakteristika u datumskim podacima:',len(date.columns))



#broj observacija u svakom setu podataka:
print('Broj observacija u numeričkim podacima:',numeric.notnull().values.sum())
print('Broj observacija u kategoričkim podacima:',category.notnull().values.sum())

print('Broj observacija u datumskim podacima:',date.notnull().values.sum())


#procenat failed products
numericmissing=numeric.isna().mean().round(4)*100
print('Procenat nedostajućih vrijednosti u numeričkim podacima:')
print(numericmissing.mean())
#data import (numeric only)
print('Procena neuspijelih produkata u numeričkim podacima: ',numeric['Response'].value_counts(normalize=True)*100)
df_train = pd.read_csv('C:/Users/Admin/Desktop/bosch-production-line-performance/train_numeric.csv.zip', nrows=100000)
num_feats = ['Id',
       'L3_S30_F3514', 'L0_S9_F200', 'L3_S29_F3430', 'L0_S11_F314',
       'L0_S0_F18', 'L3_S35_F3896', 'L0_S12_F350', 'L3_S36_F3918',
       'L0_S0_F20', 'L3_S30_F3684', 'L1_S24_F1632', 'L0_S2_F48',
       'L3_S29_F3345', 'L0_S18_F449', 'L0_S21_F497', 'L3_S29_F3433',
       'L3_S30_F3764', 'L0_S1_F24', 'L3_S30_F3554', 'L0_S11_F322',
       'L3_S30_F3564', 'L3_S29_F3327', 'L0_S2_F36', 'L0_S9_F180',
       'L3_S33_F3855', 'L0_S0_F4', 'L0_S21_F477', 'L0_S5_F114',
       'L0_S6_F122', 'L1_S24_F1122', 'L0_S9_F165', 'L0_S18_F439',
       'L1_S24_F1490', 'L0_S6_F132', 'L3_S29_F3379', 'L3_S29_F3336',
       'L0_S3_F80', 'L3_S30_F3749', 'L1_S24_F1763', 'L0_S10_F219',
 'Response'] #important features that we calculated with random forest classifier
train=df_train[num_feats] 
missing = train.isnull().sum() #number of missing values in imported data 
missing = missing[missing > 0]
missing.sort_values(inplace=True) #sorting numbers of missing values
missing.plot.bar() #ploting missing values of each features from num_feats
plt.show()
