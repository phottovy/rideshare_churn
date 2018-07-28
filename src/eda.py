import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV

#Tim Code

def load_data(filename):
    df = pd.read_csv(filename)
    return df

def df_dummies(df, dummyvars):
    for var in dummyvars:
        df = pd.concat([df, pd.get_dummies(df[var])], axis=1)
    return df

def impute_knn(df):
    impute_df = df.dropna()
    y_train = impute_df.pop('avg_rating_of_driver')
    X_train = impute_df.drop(['last_trip_date','signup_date','city','phone'],axis=1)
    k = round(sqrt(len(y_train)))

    # knn_grid = {'n_neighbors': [5,20,50,k,1000],
                # 'weights': ['uniform', 'distance'],
                # }

    # grid_knn= GridSearchCV(KNeighborsRegressor(), knn_grid)
    # kgrid = grid_knn.fit(X_train,y_train)
    # print(kgrid.best_params_)

    best_knn = KNeighborsRegressor(n_neighbors = k,weights='uniform')#,algorithm =,metric=)
    best_knn.fit(X_train, y_train)
    X_true = df.drop(['last_trip_date','signup_date','city','phone','avg_rating_of_driver'],axis=1)
    df['avg_rating_impute'] = best_knn.predict(X_true)
    df['avg_rating_of_driver'].fillna(df['avg_rating_impute'], inplace=True)

    return df

if __name__ == '__main__':
    df = pd.read_csv('data/churn_train.csv')
    df.dropna(subset=['avg_rating_by_driver'],inplace=True)

    dummyvars = ['city','phone']
    df = df_dummies(df, dummyvars)

    df_imputed = impute_knn(df)
