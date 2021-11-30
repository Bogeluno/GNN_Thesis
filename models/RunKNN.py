import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
import time

t = time.time()

# Load full Data
df_full = pd.read_csv('GNNDatasets/SimpleNNData.csv', index_col=0)
y = df_full.time_to_reservation
df_full.drop(columns=['time_to_reservation'], inplace=True)
weather_var = list(df_full.columns[-22:-1])

# Function for scores
def score_model(model, X_train, X_test, y_train, y_test):
    print(f'R2 of training: {r2_score(y_train,model.predict(X_train))}')
    print(f'R2 of test: {r2_score(y_test,model.predict(X_test))}')

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
        n_jobs=-1,
        return_train_score=True,
        verbose=10, 
        cv=cf
    )
    cv_select.fit(X_train, y_train)
    
    return(cv_select)


df = df_full.drop(columns=['index', 'hour_index']+weather_var[7:], inplace = False)


X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)


Mean_Zone_Times = dict(pd.DataFrame({'Zone': X_train.filter(regex = 'lz').idxmax(axis = 1).values, 'Time':y_train.values}).groupby('Zone').mean().squeeze())

X_train['Zone_E'] = X_train.filter(regex = 'lz').idxmax(1).map(Mean_Zone_Times)
X_test['Zone_E'] = X_test.filter(regex = 'lz').idxmax(1).map(Mean_Zone_Times)
X_train.drop(columns = X_train.filter(regex = 'lz'), inplace = True)
X_test.drop(columns =  X_test.filter(regex = 'lz'), inplace = True)


KNN_model = KNeighborsRegressor()
numerical_columns = ['leave_fuel', 'degree', 'dist_to_station','Zone_E', 'Time_Cos', 'Time_Sin']+weather_var[:7]

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


print(f'Done! Time = {time.time()-t}')