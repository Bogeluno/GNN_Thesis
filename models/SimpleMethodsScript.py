import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import time

# Load full Data
df_full = pd.read_csv('SimpleNNData.csv', index_col=0)
y = df_full.time_to_reservation
df_full.drop(columns=['time_to_reservation'], inplace=True)
weather_var = list(df_full.columns[-8:-1])

# Set up print
sys.stdout = open("SimpleResults.txt", "w")

# Function for scores
def score_model(model, X_train, X_test, y_train, y_test):
    print('')
    print(model.estimator.steps[1][1])
    print(f'R2 of training: {r2_score(y_train,model.predict(X_train))}')
    print(f'R2 of test: {r2_score(y_test,model.predict(X_test))}')
    print('----------------------------------------------')
    print('')


def getPipe(model, numerical_columns):
    """
    Prepares a pipe that 
        First:  Prepare the data prior to modelling. That is all numerical features 
                is standardized, all categotical are one.hot-encodeded. The features
                not specified as numerical or categorical are dropped if not specified.
        Second: Send the prepared data into the model.

    """
    # Pipeline to handle continous parameters. Here the parameters are scaled.
    # This is important to do each time so test data is not considered for 
    # normalization which would be the case if all data were standardized at once.
    numeric_transformer = Pipeline([
        ('scale', StandardScaler())
    ])
    
    # Split the data into continous and caterigorical using ColumnTransformer
    # and apply numeric_transformer and categorical_transformer 
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_columns)
        ],
        remainder='passthrough'
    )
    
    # Build the final pipeline for model fitting
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipe

def cv(pipe, parameters, X_train, y_train, cf = 5):    
    """
    Performs paramter tunning using cross-validation on a specifed pipe object.
    """
    # perform cross validaiton over the input parameters
    cv_select = GridSearchCV(
        estimator=pipe, 
        param_grid=parameters, 
        scoring='neg_mean_squared_error', # Use MSE
        n_jobs=-4,
        return_train_score=True,
        verbose=2, 
        cv=cf
    )
    cv_select.fit(X_train, y_train)
    
    return(cv_select)

time_start = time.time()

##################################
### NO ZONE AND LOCAL INFO ONLY
##################################
print('----------------------------------------------')
print('---NO ZONE AND LOCAL INFO ONLY')
print('----------------------------------------------')
df = df_full.drop(columns=['index', 'hour_index', 'degree', 'dist_to_station']+weather_var, inplace = False)
df.drop(columns = df.filter(regex = 'lz'), inplace = True)
numerical_columns = ['leave_fuel','Time_Cos','Time_Sin']

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1)

### LM Model
LM_model = LinearRegression(fit_intercept = True)

parameters = {}

LM_pipe = getPipe(
    model = LM_model,
    numerical_columns = numerical_columns,
)
LM_cv = cv(LM_pipe, parameters, X_train, y_train, cf = 5)

score_model(LM_cv, X_train, X_test, y_train, y_test)


### Elastic Net Model
elastic_net_model = ElasticNet(fit_intercept = True)

parameters = {
    'model__alpha': np.logspace(-3,-0.5,10),
    'model__l1_ratio': [0.001,0.01,0.1,0.5,0.9,1]
}

elastic_net_pipe = getPipe(
    model = elastic_net_model,
    numerical_columns = numerical_columns,
)
elastic_net_cv = cv(elastic_net_pipe, parameters, X_train, y_train, cf = 5)

best_idx = elastic_net_cv.cv_results_['mean_test_score'].argmax()
alpha = elastic_net_cv.cv_results_['param_model__alpha'].data[best_idx]
l1_ratio = elastic_net_cv.cv_results_['param_model__l1_ratio'].data[best_idx]

print(f'alpha = {round(alpha, 3)}, l1-ratio: {round(l1_ratio,3)}')
score_model(elastic_net_cv, X_train, X_test, y_train, y_test)


### RF 
RF_model = RandomForestRegressor()

parameters = {
    'model__n_estimators': [100,200,300,400],
    'model__min_samples_leaf': [10,20,30]
}

RF_pipe = getPipe(
    model = RF_model,
    numerical_columns = numerical_columns,
)
RF_cv = cv(RF_pipe, parameters, X_train, y_train, cf = 5)

best_idx = RF_cv.cv_results_['mean_test_score'].argmax()
n_estimators = RF_cv.cv_results_['param_model__n_estimators'].data[best_idx]
min_samples_leaf = RF_cv.cv_results_['param_model__min_samples_leaf'].data[best_idx]

print(f'n_estimators = {n_estimators}', f'min_samples_leaf = {min_samples_leaf}')

score_model(RF_cv, X_train, X_test, y_train, y_test)


### KNN
KNN_model = KNeighborsRegressor()

parameters = {
    'model__n_neighbors': [50,100,200,300],
    'model__weights': ['uniform', 'distance']
}

KNN_pipe = getPipe(
    model = KNN_model,
    numerical_columns = numerical_columns,
)
KNN_cv = cv(KNN_pipe, parameters, X_train, y_train, cf = 5)


best_idx = KNN_cv.cv_results_['mean_test_score'].argmax()
n_neighbors = KNN_cv.cv_results_['param_model__n_neighbors'].data[best_idx]
weights = KNN_cv.cv_results_['param_model__weights'].data[best_idx]

print(f'n_neighbors = {n_neighbors}', f'weights = {weights}')

score_model(KNN_cv, X_train, X_test, y_train, y_test)


print(f'Time spent: {time.time()-time_start}')
print('\n\n')
##################################
### Add non-local info
##################################
print('----------------------------------------------')
print('--ADD NON-LOCAL INFO')
print('----------------------------------------------')
df = df_full.drop(columns=['index', 'hour_index']+weather_var, inplace = False)
df.drop(columns = df.filter(regex = 'lz'), inplace = True)
numerical_columns = ['leave_fuel', 'Time_Cos','Time_Sin','degree', 'dist_to_station']

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1)


### LM Model
LM_model = LinearRegression(fit_intercept = True)

parameters = {}

LM_pipe = getPipe(
    model = LM_model,
    numerical_columns = numerical_columns,
)
LM_cv = cv(LM_pipe, parameters, X_train, y_train, cf = 5)

score_model(LM_cv, X_train, X_test, y_train, y_test)


### Elastic Net
elastic_net_model = ElasticNet(fit_intercept = True)

parameters = {
    'model__alpha': np.logspace(-3, -0.5, 10),
    'model__l1_ratio': [0.01,0.05,0.1,0.2]
}

elastic_net_pipe = getPipe(
    model = elastic_net_model,
    numerical_columns = numerical_columns,
)
elastic_net_cv = cv(elastic_net_pipe, parameters, X_train, y_train, cf = 5)


best_idx = elastic_net_cv.cv_results_['mean_test_score'].argmax()
alpha = elastic_net_cv.cv_results_['param_model__alpha'].data[best_idx]
l1_ratio = elastic_net_cv.cv_results_['param_model__l1_ratio'].data[best_idx]

print(f'alpha = {round(alpha, 3)}, l1-ratio: {round(l1_ratio,3)}')
score_model(elastic_net_cv, X_train, X_test, y_train, y_test)

### RF 
RF_model = RandomForestRegressor()

parameters = {
    'model__n_estimators': [100,200, 300, 400],
    'model__min_samples_leaf': [10,20,30]
}

RF_pipe = getPipe(
    model = RF_model,
    numerical_columns = numerical_columns,
)
RF_cv = cv(RF_pipe, parameters, X_train, y_train, cf = 5)

best_idx = RF_cv.cv_results_['mean_test_score'].argmax()
n_estimators = RF_cv.cv_results_['param_model__n_estimators'].data[best_idx]
min_samples_leaf = RF_cv.cv_results_['param_model__min_samples_leaf'].data[best_idx]

print(f'n_estimators = {n_estimators}', f'min_samples_leaf = {min_samples_leaf}')

score_model(RF_cv, X_train, X_test, y_train, y_test)


### KNN-model
KNN_model = KNeighborsRegressor()

parameters = {
    'model__n_neighbors': [50,100,200],
    'model__weights': ['uniform', 'distance']
}

KNN_pipe = getPipe(
    model = KNN_model,
    numerical_columns = numerical_columns,
)
KNN_cv = cv(KNN_pipe, parameters, X_train, y_train, cf = 5)

best_idx = KNN_cv.cv_results_['mean_test_score'].argmax()
n_neighbors = KNN_cv.cv_results_['param_model__n_neighbors'].data[best_idx]
weights = KNN_cv.cv_results_['param_model__weights'].data[best_idx]
 
print(f'n_neighbors = {n_neighbors}', f'weights = {weights}')
score_model(KNN_cv, X_train, X_test, y_train, y_test)


print(f'Time spent: {time.time()-time_start}')
print('\n\n')
##################################
### Add weather
##################################
print('----------------------------------------------')
print('--ADD WEATHER')
print('----------------------------------------------')
df = df_full.drop(columns=['index', 'hour_index'], inplace = False)
df.drop(columns = df.filter(regex = 'lz'), inplace = True)
numerical_columns = ['leave_fuel', 'degree', 'dist_to_station','Time_Cos','Time_Sin']+weather_var

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1)


### LM Model
LM_model = LinearRegression(fit_intercept = True)

parameters = {}

LM_pipe = getPipe(
    model = LM_model,
    numerical_columns = numerical_columns,
)
LM_cv = cv(LM_pipe, parameters, X_train, y_train, cf = 5)

score_model(LM_cv, X_train, X_test, y_train, y_test)


### Elastic Net
elastic_net_model = ElasticNet(fit_intercept = True)

parameters = {
    'model__alpha': np.logspace(-3, -0.5, 10),
    'model__l1_ratio': [0.01,0.1,0.2,0.5,0.8,1]
}

elastic_net_pipe = getPipe(
    model = elastic_net_model,
    numerical_columns = numerical_columns,
)
elastic_net_cv = cv(elastic_net_pipe, parameters, X_train, y_train, cf = 5)


best_idx = elastic_net_cv.cv_results_['mean_test_score'].argmax()
alpha = elastic_net_cv.cv_results_['param_model__alpha'].data[best_idx]
l1_ratio = elastic_net_cv.cv_results_['param_model__l1_ratio'].data[best_idx]

print(f'alpha = {round(alpha, 3)}, l1-ratio: {round(l1_ratio,3)}')
score_model(elastic_net_cv, X_train, X_test, y_train, y_test)

### RF 
RF_model = RandomForestRegressor()

parameters = {
    'model__n_estimators': [100,200,300,400],
    'model__min_samples_leaf': [10,20,30]
}

RF_pipe = getPipe(
    model = RF_model,
    numerical_columns = numerical_columns,
)
RF_cv = cv(RF_pipe, parameters, X_train, y_train, cf = 5)

best_idx = RF_cv.cv_results_['mean_test_score'].argmax()
n_estimators = RF_cv.cv_results_['param_model__n_estimators'].data[best_idx]
min_samples_leaf = RF_cv.cv_results_['param_model__min_samples_leaf'].data[best_idx]

print(f'n_estimators = {n_estimators}', f'min_samples_leaf = {min_samples_leaf}')

score_model(RF_cv, X_train, X_test, y_train, y_test)


### KNN-model
KNN_model = KNeighborsRegressor()

parameters = {
    'model__n_neighbors': [50,100,150,200],
    'model__weights': ['uniform', 'distance']
}

KNN_pipe = getPipe(
    model = KNN_model,
    numerical_columns = numerical_columns,
)
KNN_cv = cv(KNN_pipe, parameters, X_train, y_train, cf = 5)

best_idx = KNN_cv.cv_results_['mean_test_score'].argmax()
n_neighbors = KNN_cv.cv_results_['param_model__n_neighbors'].data[best_idx]
weights = KNN_cv.cv_results_['param_model__weights'].data[best_idx]
 
print(f'n_neighbors = {n_neighbors}', f'weights = {weights}')
score_model(KNN_cv, X_train, X_test, y_train, y_test)


print(f'Time spent: {time.time()-time_start}')
print('\n\n')


##################################
### Add zones
##################################
print('----------------------------------------------')
print('--ADD ZONES')
print('----------------------------------------------')
df = df_full.drop(columns=['index', 'hour_index'], inplace = False)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1)

numerical_columns = ['leave_fuel', 'degree', 'dist_to_station', 'Time_Cos', 'Time_Sin']+weather_var

### LM Model
LM_model = LinearRegression(fit_intercept = True)

parameters = {}

LM_pipe = getPipe(
    model = LM_model,
    numerical_columns = numerical_columns,
)
LM_cv = cv(LM_pipe, parameters, X_train, y_train, cf = 5)

score_model(LM_cv, X_train, X_test, y_train, y_test)


### Elastic Net
elastic_net_model = ElasticNet(fit_intercept = True)
numerical_columns = ['leave_fuel', 'degree', 'dist_to_station', 'Time_Cos', 'Time_Sin']+weather_var[:7]

parameters = {
    'model__alpha': np.logspace(-3, 0, 10),
    'model__l1_ratio': [0.1,0.5,0.9,1]
}

elastic_net_pipe = getPipe(
    model = elastic_net_model,
    numerical_columns = numerical_columns,
)
elastic_net_cv = cv(elastic_net_pipe, parameters, X_train, y_train, cf = 5)

best_idx = elastic_net_cv.cv_results_['mean_test_score'].argmax()
alpha = elastic_net_cv.cv_results_['param_model__alpha'].data[best_idx]
l1_ratio = elastic_net_cv.cv_results_['param_model__l1_ratio'].data[best_idx]

print(f'alpha = {round(alpha, 3)}, l1-ratio: {round(l1_ratio,3)}')

score_model(elastic_net_cv, X_train, X_test, y_train, y_test)


print(f'Time spent: {time.time()-time_start}')
print('\n\n')

##################################
### Encoded Zones
##################################
print('----------------------------------------------')
print('--ENCODE ZONES')
print('----------------------------------------------')
df = df_full.drop(columns=['index', 'hour_index'], inplace = False)
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1)

# Zone encoding
Mean_Zone_Times = dict(pd.DataFrame({'Zone': X_train.filter(regex = 'lz').idxmax(axis = 1).values, 'Time':y_train.values}).groupby('Zone').mean().squeeze())

X_train['Zone_E'] = X_train.filter(regex = 'lz').idxmax(1).map(Mean_Zone_Times)
X_test['Zone_E'] = X_test.filter(regex = 'lz').idxmax(1).map(Mean_Zone_Times)
X_train.drop(columns = X_train.filter(regex = 'lz'), inplace = True)
X_test.drop(columns =  X_test.filter(regex = 'lz'), inplace = True)

numerical_columns = ['leave_fuel', 'degree', 'dist_to_station','Zone_E', 'Time_Cos', 'Time_Sin']+weather_var

### LM Model
LM_model = LinearRegression(fit_intercept = False)


parameters = {}

LM_pipe = getPipe(
    model = LM_model,
    numerical_columns = numerical_columns,
)
LM_cv = cv(LM_pipe, parameters, X_train, y_train, cf = 5)

score_model(LM_cv, X_train, X_test, y_train, y_test)


### Elastic Net
elastic_net_model = ElasticNet(fit_intercept = True)

parameters = {
    'model__alpha': np.logspace(-3, -1, 10),
    'model__l1_ratio': [0.001,0.01,0.1]
}

elastic_net_pipe = getPipe(
    model = elastic_net_model,
    numerical_columns = numerical_columns,
)
elastic_net_cv = cv(elastic_net_pipe, parameters, X_train, y_train, cf = 5)

best_idx = elastic_net_cv.cv_results_['mean_test_score'].argmax()
alpha = elastic_net_cv.cv_results_['param_model__alpha'].data[best_idx]
l1_ratio = elastic_net_cv.cv_results_['param_model__l1_ratio'].data[best_idx]

print(f'alpha = {round(alpha, 3)}, l1-ratio: {round(l1_ratio,3)}')
score_model(elastic_net_cv, X_train, X_test, y_train, y_test)


### RF 
RF_model = RandomForestRegressor()

parameters = {
    'model__n_estimators': [100,200,300,400],
    'model__min_samples_leaf': [5,10,20]
}

RF_pipe = getPipe(
    model = RF_model,
    numerical_columns = numerical_columns,
)
RF_cv = cv(RF_pipe, parameters, X_train, y_train, cf = 5)

best_idx = RF_cv.cv_results_['mean_test_score'].argmax()
n_estimators = RF_cv.cv_results_['param_model__n_estimators'].data[best_idx]
min_samples_leaf = RF_cv.cv_results_['param_model__min_samples_leaf'].data[best_idx]

print(f'n_estimators = {n_estimators}', f'min_samples_leaf = {min_samples_leaf}')
score_model(RF_cv, X_train, X_test, y_train, y_test)


### KNN model
KNN_model = KNeighborsRegressor()

parameters = {
    'model__n_neighbors': [20,35, 50, 100],
    'model__weights': ['uniform', 'distance']
}

KNN_pipe = getPipe(
    model = KNN_model,
    numerical_columns = numerical_columns,
)
KNN_cv = cv(KNN_pipe, parameters, X_train, y_train, cf = 5)

best_idx = KNN_cv.cv_results_['mean_test_score'].argmax()
n_neighbors = KNN_cv.cv_results_['param_model__n_neighbors'].data[best_idx]
weights = KNN_cv.cv_results_['param_model__weights'].data[best_idx]
 
print(f'n_neighbors = {n_neighbors}', f'weights = {weights}')
score_model(KNN_cv, X_train, X_test, y_train, y_test)


print(f'Time spent: {time.time()-time_start}')
print('\n\n')

##################################
### End and exit stdout
##################################
sys.stdout.close()