import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, classification_report
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import time

loops = 5

df_full = pd.read_csv('Data/SimpleNNData.csv', index_col=0, parse_dates = [1]).sort_values(by = 'time')
y = df_full.time_to_reservation

# Load weather
Weather_Scale = pd.read_csv('Data/MinMaxWeather.csv', index_col=0)
weather_var = list(Weather_Scale.index)

# Load slicing
with open("Data/Sample_CC", "rb") as fp: 
    cc = pickle.load(fp)
train_idx = np.concatenate(cc[:2])
test_idx = cc[2]

# Set up print
sys.stdout = open("Results/RFResults.txt", "w")

def cv(pipe, parameters, X_train, y_train, cf = 4):    
    """
    Performs paramter tunning using cross-validation on a specifed pipe object.
    """
    # perform cross validaiton over the input parameters
    cv_select = GridSearchCV(
        estimator=pipe, 
        param_grid=parameters, 
        scoring='r2', 
        n_jobs=-4,
        return_train_score=True,
        verbose=0, 
        cv=cf
    )
    cv_select.fit(X_train, y_train)
    
    return(cv_select)

with open("Data/Sample_NC", "rb") as fp: 
    nc = pickle.load(fp)

# For classification
Clas_Coef = dict(pd.concat([df_full.time.dt.hour.iloc[np.concatenate(cc[:2])],df_full.time_to_reservation.iloc[np.concatenate(cc[:2])]], axis = 1).groupby('time')['time_to_reservation'].mean()*2)
df_clas = pd.concat([df_full.time.dt.hour.iloc[cc[2]],df_full.time_to_reservation.iloc[cc[2]]], axis = 1)
df_clas['Cut'] = df_clas.time.map(dict(Clas_Coef))
df_full.drop(columns=['time_to_reservation', 'hour_index', 'time'], inplace=True)

# Normalization
df_full['leave_fuel'] = df_full['leave_fuel']/100
df_full['degree'] = df_full['degree']/50
df_full['dist_to_station'] = df_full['dist_to_station']/5000
df_full[weather_var] = (df_full[weather_var] - Weather_Scale['Min'])/Weather_Scale['diff']

# Function for scores
def score_model(model, X_train, X_test, y_train, y_test, df_clas = df_clas):
    preds = model.predict(X_test)
    print('')
    print(f'R2 of training: {r2_score(y_train,model.predict(X_train))}')
    print(f'R2 of test: {r2_score(y_test,preds)}')
    df_clas['Preds'] = preds
    print('F1-score:',classification_report(df_clas.time_to_reservation > df_clas.Cut, df_clas.Preds > df_clas.Cut, target_names = ['Under','Over'], zero_division = 0, output_dict = True)['Over']['f1-score'])
    print('----------------------------------------------')
    print('')
    print('')

time_start = time.time()

##################################
### NO ZONES
##################################
print('----------------------------------------------')
print('---NO ZONES')
print('----------------------------------------------')

# Prep data
df = df_full.drop(columns = list(df_full.filter(regex = 'lz').columns) + weather_var + ['dist_to_station'])
X_train = df.iloc[train_idx]
y_train = y.iloc[train_idx]
X_test = df.iloc[test_idx]
y_test = y.iloc[test_idx]

parameters = {
        'n_estimators': [100,200,300,400,500],
        'min_samples_leaf': [5,10,20,30]
        }

for _ in tqdm(range(loops)):
    RF_model = RandomForestRegressor()

    RF_cv = cv(RF_model, parameters, X_train, y_train, cf = 4)

    best_idx = RF_cv.cv_results_['mean_test_score'].argmax()
    n_estimators = RF_cv.cv_results_['param_n_estimators'].data[best_idx]
    min_samples_leaf = RF_cv.cv_results_['param_min_samples_leaf'].data[best_idx]

    print(RF_cv.estimator)
    print(f'n_estimators = {n_estimators}', f'min_samples_leaf = {min_samples_leaf}')
    score_model(RF_cv, X_train, X_test, y_train, y_test)

print(f'Time spent: {time.time()-time_start}')
print('\n\n')


##################################
### ADD ZONES
##################################
print('----------------------------------------------')
print('---ADD ZONES')
print('----------------------------------------------')

# Prep data
df = df_full.drop(columns =  weather_var + ['dist_to_station'])
X_train = df.iloc[train_idx]
y_train = y.iloc[train_idx]
X_test = df.iloc[test_idx]
y_test = y.iloc[test_idx]


for _ in tqdm(range(loops)):
    RF_model = RandomForestRegressor()

    RF_cv = cv(RF_model, parameters, X_train, y_train, cf = 4)

    best_idx = RF_cv.cv_results_['mean_test_score'].argmax()
    n_estimators = RF_cv.cv_results_['param_n_estimators'].data[best_idx]
    min_samples_leaf = RF_cv.cv_results_['param_min_samples_leaf'].data[best_idx]

    print(RF_cv.estimator)
    print(f'n_estimators = {n_estimators}', f'min_samples_leaf = {min_samples_leaf}')
    score_model(RF_cv, X_train, X_test, y_train, y_test)


print(f'Time spent: {time.time()-time_start}')
print('\n\n')


##################################
### ADD ENCODED ZONES
##################################
print('----------------------------------------------')
print('---ADD ENCODED ZONES')
print('----------------------------------------------')

# Prep data
df = df_full.drop(columns =  weather_var + ['dist_to_station'])
X_train = df.iloc[train_idx]
y_train = y.iloc[train_idx]
X_test = df.iloc[test_idx]
y_test = y.iloc[test_idx]

# Encode zones
Mean_Zone_Times = dict(pd.DataFrame({'Zone': X_train.filter(regex = 'lz').idxmax(axis = 1).values, 'Time':y_train.values}).groupby('Zone').mean().squeeze())

X_train['Zone_E'] = X_train.filter(regex = 'lz').idxmax(1).map(Mean_Zone_Times)
X_test['Zone_E'] = X_test.filter(regex = 'lz').idxmax(1).map(Mean_Zone_Times)
X_train.drop(columns = X_train.filter(regex = 'lz'), inplace = True)
X_test.drop(columns =  X_test.filter(regex = 'lz'), inplace = True)

for _ in tqdm(range(loops)):
    RF_model = RandomForestRegressor()

    parameters = {
        'n_estimators': [100,200,300,400,500],
        'min_samples_leaf': [5,10,20,30]
    }

    RF_cv = cv(RF_model, parameters, X_train, y_train, cf = 4)

    best_idx = RF_cv.cv_results_['mean_test_score'].argmax()
    n_estimators = RF_cv.cv_results_['param_n_estimators'].data[best_idx]
    min_samples_leaf = RF_cv.cv_results_['param_min_samples_leaf'].data[best_idx]

    print(RF_cv.estimator)
    print(f'n_estimators = {n_estimators}', f'min_samples_leaf = {min_samples_leaf}')
    score_model(RF_cv, X_train, X_test, y_train, y_test)


print(f'Time spent: {time.time()-time_start}')
print('\n\n')
##################################
### ADD WEATHER AND DIST
##################################
print('----------------------------------------------')
print('---ADD WEATHER AND DIST')
print('----------------------------------------------')
# Prep data
df = df_full.drop(columns = list(df_full.filter(regex = 'lz').columns))
X_train = df.iloc[train_idx]
y_train = y.iloc[train_idx]
X_test = df.iloc[test_idx]
y_test = y.iloc[test_idx]

for _ in tqdm(range(loops)):
    RF_model = RandomForestRegressor()

    RF_cv = cv(RF_model, parameters, X_train, y_train, cf = 4)

    best_idx = RF_cv.cv_results_['mean_test_score'].argmax()
    n_estimators = RF_cv.cv_results_['param_n_estimators'].data[best_idx]
    min_samples_leaf = RF_cv.cv_results_['param_min_samples_leaf'].data[best_idx]

    print(RF_cv.estimator)
    print(f'n_estimators = {n_estimators}', f'min_samples_leaf = {min_samples_leaf}')
    score_model(RF_cv, X_train, X_test, y_train, y_test)



print(f'Time spent: {time.time()-time_start}')
print('\n\n')
##################################
### With all
##################################
print('----------------------------------------------')
print('---WITH ALL')
print('----------------------------------------------')
# Prep data
df = df_full.copy()
X_train = df.iloc[train_idx]
y_train = y.iloc[train_idx]
X_test = df.iloc[test_idx]
y_test = y.iloc[test_idx]

for _ in tqdm(range(loops)):
    RF_model = RandomForestRegressor()

    RF_cv = cv(RF_model, parameters, X_train, y_train, cf = 4)

    best_idx = RF_cv.cv_results_['mean_test_score'].argmax()
    n_estimators = RF_cv.cv_results_['param_n_estimators'].data[best_idx]
    min_samples_leaf = RF_cv.cv_results_['param_min_samples_leaf'].data[best_idx]

    print(RF_cv.estimator)
    print(f'n_estimators = {n_estimators}', f'min_samples_leaf = {min_samples_leaf}')
    score_model(RF_cv, X_train, X_test, y_train, y_test)


print(f'Time spent: {time.time()-time_start}')
print('\n\n')

##################################
### With all and encoded
##################################
print('----------------------------------------------')
print('---WITH ALL AND ENCODED')
print('----------------------------------------------')
# Prep data
df = df_full.copy()
X_train = df.iloc[train_idx]
y_train = y.iloc[train_idx]
X_test = df.iloc[test_idx]
y_test = y.iloc[test_idx]

# Encode zones
Mean_Zone_Times = dict(pd.DataFrame({'Zone': X_train.filter(regex = 'lz').idxmax(axis = 1).values, 'Time':y_train.values}).groupby('Zone').mean().squeeze())

X_train['Zone_E'] = X_train.filter(regex = 'lz').idxmax(1).map(Mean_Zone_Times)
X_test['Zone_E'] = X_test.filter(regex = 'lz').idxmax(1).map(Mean_Zone_Times)
X_train.drop(columns = X_train.filter(regex = 'lz'), inplace = True)
X_test.drop(columns =  X_test.filter(regex = 'lz'), inplace = True)

for _ in tqdm(range(loops)):
    RF_model = RandomForestRegressor()

    RF_cv = cv(RF_model, parameters, X_train, y_train, cf = 4)

    best_idx = RF_cv.cv_results_['mean_test_score'].argmax()
    n_estimators = RF_cv.cv_results_['param_n_estimators'].data[best_idx]
    min_samples_leaf = RF_cv.cv_results_['param_min_samples_leaf'].data[best_idx]

    print(RF_cv.estimator)
    print(f'n_estimators = {n_estimators}', f'min_samples_leaf = {min_samples_leaf}')
    score_model(RF_cv, X_train, X_test, y_train, y_test)


print(f'Time spent: {time.time()-time_start}')

##################################
### End and exit stdout
##################################
sys.stdout.close()