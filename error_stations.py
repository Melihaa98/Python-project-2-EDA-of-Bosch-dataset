import numpy as np 
import pandas as pd 
import seaborn as sns

import matplotlib.pyplot as plt
#import data
numeric= 'C:/Users/Admin/Desktop/bosch-production-line-performance/train_numeric.csv.zip'
features = pd.read_csv(numeric, nrows=1).drop(['Response', 'Id'], axis=1).columns.values #using all columns for features except prediction target

#function isolate all the features that belong to specific Lines and Stations. 
def orgainize(features):
    line_features = {}
    station_features = {}
    lines = set([f.split('_')[0] for f in features])
    stations = set([f.split('_')[1] for f in features])
    
    for l in lines:
        line_features[l] = [f for f in features if l+'_' in f]
        
    for s in stations:
        station_features[s] = [f for f in features if s+'_' in f]
        
            
    return line_features, station_features

line_features, station_features = orgainize(features)
print("Features in Station 32: {}".format( station_features['S32'] ))
#to know if certain stations were correlated to higher error rates
#using for loop to go through all stations and calculate error of each one
#Features - Total features in the Station.
#Samples - Total samples with measured values (non-NaN) >= 1 in the Station
#Error rate - (Response==1) rate for samples in the Station.    

station_error = []
for s in station_features:
    cols = ['Id', 'Response']
    cols.extend(station_features[s])
    df = pd.read_csv(numeric, usecols=cols).dropna(subset=station_features[s], how='all')
    error_rate = df[df.Response == 1].size / float(df[df.Response == 0].size)
    station_error.append([df.shape[1]-2, df.shape[0], error_rate]) 
    
station_data = pd.DataFrame(station_error, 
                         columns=['Features', 'Samples', 'Error_Rate'], 
                         index=station_features).sort_index()
print(station_data)
plt.figure(figsize=(8, 20))
sns.barplot(x='Error_Rate', y=station_data.index.values, data=station_data, color="red")
plt.title('Stopa pogreške između stanica')

plt.xlabel('Stopa pogreške stanice')
plt.show() #ploting error rate of each station
