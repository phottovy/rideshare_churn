import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.neighbors import KNeighborsRegressor



def load_churn_data(filename):
    df = pd.read_csv(filename)
    df.dropna(subset=['avg_rating_by_driver'], inplace=True)
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'], yearfirst=True, format='%Y/%m/%d')
    df['signup_date'] = pd.to_datetime(df['signup_date'], yearfirst=True, format='%Y/%m/%d')
    churn_date = pd.datetime(2014, 7, 1) - pd.to_timedelta(30, unit='D')
    two_week_date = pd.datetime(2014, 7, 1) - pd.to_timedelta(14, unit='D')
    df['ride_last_two_weeks'] = df["last_trip_date"] >= two_week_date
    df['churn'] = df["last_trip_date"] < churn_date
    return df

def df_dummies(df, dummyvars):
    for var in dummyvars:
        df = pd.concat([df, pd.get_dummies(df[var])], axis=1)
    return df

def create_Xy(df, y_col, X_drop):
    y = df.pop(y_col)
    X = df.drop(X_drop, axis=1)
    return X, y

def impute_knn(df):
    impute_df = df.dropna()
    X_impute, y_impute = create_Xy(impute_df, 'avg_rating_of_driver', [
                                   'last_trip_date', 'signup_date', 'city', 'phone'])
    k = round(sqrt(len(y_impute)))

    # knn_grid = {'n_neighbors': [5,20,50,k,1000],
                # 'weights': ['uniform', 'distance'],
                # }

    # grid_knn= GridSearchCV(KNeighborsRegressor(), knn_grid)
    # kgrid = grid_knn.fit(X_impute,y_impute)
    # print(kgrid.best_params_)

    best_knn = KNeighborsRegressor(n_neighbors=k, weights='uniform')
    best_knn.fit(X_impute, y_impute)
    X_true = df.drop(['last_trip_date', 'signup_date', 'city',
                      'phone', 'avg_rating_of_driver'], axis=1)
    df['avg_rating_impute'] = best_knn.predict(X_true)
    df['avg_rating_of_driver'].fillna(df['avg_rating_impute'], inplace=True)
    df.drop(['avg_rating_impute'], axis=1, inplace=True)
    return df


if __name__ == '__main__':
    # load data
    df = load_churn_data('../data/churn_train.csv')

    # create dummies
    dummyvars = ['city', 'phone']
    df = df_dummies(df, dummyvars)

    #impute avg_rating_of_driver
    df_imputed = impute_knn(df)

    #Create X and y arrays
    Xdf, ydf = create_Xy(df_imputed, 'churn', ['last_trip_date', 'signup_date', 'city', 'phone'])
    X = Xdf.values
    y = ydf.values
    Xcols = Xdf.columns

    # create train test split on the training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # standardize data
    standardizer = StandardScaler()
    standardizer.fit(X_train, y_train)
    X_train_std = standardizer.transform(X_train)
    X_test_std = standardizer.transform(X_test)

    # create scoring list for classification models
    #scoring = ['roc_auc', 'accuracy']


    # random forest grid search

    rf_grid = {'max_depth': [10, None],
               'min_samples_split': [2, 4],
               'min_samples_leaf': [1, 2],
               'n_estimators': [100],
               'random_state': [1]}

    rf_gridsearch = GridSearchCV(RandomForestClassifier(),
                                 rf_grid,
                                 n_jobs=-1,
                                 verbose=True,
                                 cv=3)

    rf_gridsearch.fit(X_train_std, y_train)

    print("best parameters rf: {}".format(rf_gridsearch.best_params_))

    rf_best_model = rf_gridsearch.best_estimator_

    rf_best_model.fit(X_train_std, y_train)

    rf_y_preds_best = rf_best_model.predict(X_test_std)

    rf_best_roc_auc = roc_auc_score(y_test, rf_y_preds_best)
    rf_best_accuracy = accuracy_score(y_test, rf_y_preds_best)
    print('best rf model roc_auc: {}'.format(rf_best_roc_auc))
    print('best rf model accuracy: {}'.format(rf_best_accuracy))

    # adaboost grid search

    ab_grid = {'n_estimators': [50, 100],
               'random_state': [1],
               'learning_rate': [0.05]}

    ab_gridsearch = GridSearchCV(AdaBoostClassifier(),
                                 ab_grid,
                                 n_jobs=-1,
                                 verbose=True,
                                 cv=3)

    ab_gridsearch.fit(X_train_std, y_train)

    print("best parameters ab: {}".format(ab_gridsearch.best_params_))

    ab_best_model = ab_gridsearch.best_estimator_

    ab_best_model.fit(X_train_std, y_train)

    ab_y_preds_best = ab_best_model.predict(X_test_std)

    ab_best_roc_auc = roc_auc_score(y_test, ab_y_preds_best)
    ab_best_accuracy = accuracy_score(y_test, ab_y_preds_best)
    print('best ab model roc_auc: {}'.format(ab_best_roc_auc))
    print('best ab model accuracy: {}'.format(ab_best_accuracy))

    # gradient boosing grid search

    gdbr_grid = {'max_depth': [10],
                 'min_samples_split': [4],
                 'min_samples_leaf': [1],
                 'n_estimators': [100],
                 'random_state': [1],
                 'learning_rate': [0.05]}

    gdbr_gridsearch = GridSearchCV(GradientBoostingClassifier(),
                                   gdbr_grid,
                                   n_jobs=-1,
                                   verbose=True,
                                   cv=3)

    gdbr_gridsearch.fit(X_train_std, y_train)

    print("best parameters gdbr: {}".format(gdbr_gridsearch.best_params_))

    gdbr_best_model = gdbr_gridsearch.best_estimator_

    gdbr_best_model.fit(X_train_std, y_train)

    gdbr_y_preds_best = gdbr_best_model.predict(X_test_std)

    gdbr_best_roc_auc = roc_auc_score(y_test, gdbr_y_preds_best)
    gdbr_best_accuracy = accuracy_score(y_test, gdbr_y_preds_best)
    print('best gdbr model roc_auc: {}'.format(gdbr_best_roc_auc))
    print('best gdbr model accuracy: {}'.format(gdbr_best_accuracy))

    # decision tree grid search

    dt_grid = {'max_depth': [10, None],
               'min_samples_split': [2, 4],
               'min_samples_leaf': [1, 2],
               'random_state': [1]}

    dt_gridsearch = GridSearchCV(DecisionTreeClassifier(),
                                 dt_grid,
                                 n_jobs=-1,
                                 verbose=True,
                                 cv=3)

    dt_gridsearch.fit(X_train_std, y_train)

    print("best parameters dt: {}".format(dt_gridsearch.best_params_))

    dt_best_model = dt_gridsearch.best_estimator_

    dt_best_model.fit(X_train_std, y_train)

    dt_y_preds_best = dt_best_model.predict(X_test_std)

    dt_best_roc_auc = roc_auc_score(y_test, dt_y_preds_best)
    dt_best_accuracy = accuracy_score(y_test, dt_y_preds_best)
    print('best dt model roc_auc: {}'.format(dt_best_roc_auc))
    print('best dt model accuracy: {}'.format(dt_best_accuracy))

    # logistic regression grid search

    log_grid = {'random_state': [1]}

    log_gridsearch = GridSearchCV(LogisticRegression(),
                                  log_grid,
                                  n_jobs=-1,
                                  verbose=True,
                                  cv=3)

    log_gridsearch.fit(X_train_std, y_train)

    print("best parameters dt: {}".format(log_gridsearch.best_params_))

    log_best_model = log_gridsearch.best_estimator_

    log_best_model.fit(X_train_std, y_train)

    log_y_preds_best = log_best_model.predict(X_test_std)

    log_best_roc_auc = roc_auc_score(y_test, log_y_preds_best)
    log_best_accuracy = accuracy_score(y_test, log_y_preds_best)
    print('best log reg model roc_auc: {}'.format(log_best_roc_auc))
    print('best log reg model accuracy: {}'.format(log_best_accuracy))

    ### RUN BEST MODEL ON TEST DATA ###

    # load data
    df_final = load_churn_data('../data/churn_test.csv')

    # create dummies
    dummyvars = ['city', 'phone']
    df_final = df_dummies(df_final, dummyvars)

    #impute avg_rating_of_driver
    df_imputed_final = impute_knn(df_final)

    #Create X and y arrays
    Xdf_final, ydf_final = create_Xy(df_imputed_final, 'churn', [
                                     'last_trip_date', 'signup_date', 'city', 'phone'])
    X_final = Xdf.values
    y_final = ydf.values
    Xcols = Xdf_final.columns

    # standardize data
    standardizer_final = StandardScaler()
    standardizer_final.fit(X_final, y_final)
    X_final_std = standardizer.transform(X_final)

    # random forest final run
    rf_y_preds_best_final = rf_best_model.predict(X_final_std)
    rf_y_probs_final = rf_best_model.predict_proba(X_final_std)

    rf_best_roc_auc_final = roc_auc_score(y_final, rf_y_preds_best_final)
    rf_best_accuracy_final = accuracy_score(y_final, rf_y_preds_best_final)
    print('final rf model roc_auc: {}'.format(rf_best_roc_auc_final))
    print('final rf model accuracy: {}'.format(rf_best_accuracy_final))

    # ada boost final run

    ab_y_preds_best_final = ab_best_model.predict(X_final_std)
    ab_y_probs_final = ab_best_model.predict_proba(X_final_std)

    ab_best_roc_auc_final = roc_auc_score(y_final, ab_y_preds_best_final)
    ab_best_accuracy_final = accuracy_score(y_final, ab_y_preds_best_final)
    print('final ab model roc_auc: {}'.format(ab_best_roc_auc_final))
    print('final ab model accuracy: {}'.format(ab_best_accuracy_final))

    # gradient boosting final run

    gdbr_y_preds_best_final = gdbr_best_model.predict(X_final_std)
    gdbr_y_probs_final = gdbr_best_model.predict_proba(X_final_std)


    gdbr_best_roc_auc_final = roc_auc_score(y_final, gdbr_y_preds_best_final)
    gdbr_best_accuracy_final = accuracy_score(y_final, gdbr_y_preds_best_final)
    print('final gdbr model roc_auc: {}'.format(gdbr_best_roc_auc_final))
    print('final gdbr model accuracy: {}'.format(gdbr_best_accuracy_final))

    # decision tree final run

    dt_y_preds_best_final = dt_best_model.predict(X_final_std)
    dt_y_probs_final = dt_best_model.predict_proba(X_final_std)

    dt_best_roc_auc_final = roc_auc_score(y_final, dt_y_preds_best_final)
    dt_best_accuracy_final = accuracy_score(y_final, dt_y_preds_best_final)
    print('final dt model roc_auc: {}'.format(dt_best_roc_auc_final))
    print('final dt model accuracy: {}'.format(dt_best_accuracy_final))

    # logistic regression final run

    log_y_preds_best_final = log_best_model.predict(X_final_std)
    log_y_probs_final = log_best_model.predict_proba(X_final_std)


    log_best_roc_auc_final = roc_auc_score(y_final, log_y_preds_best_final)
    log_best_accuracy_final = accuracy_score(y_final, log_y_preds_best_final)
    print('final log reg model roc_auc: {}'.format(log_best_roc_auc_final))
    print('final log reg model accuracy: {}'.format(log_best_accuracy_final))

    ### Plots for presentation ###

    # roc curve

    rf_fpr, rf_tpr, t1 = roc_curve(y_final, rf_y_probs_final[:, 1])
    ab_fpr, ab_tpr, t2 = roc_curve(y_final, ab_y_probs_final[:, 1])
    gdbr_fpr, gdbr_tpr, t3 = roc_curve(y_final, gdbr_y_probs_final[:, 1])
    dt_fpr, dt_tpr, t4 = roc_curve(y_final, dt_y_probs_final[:, 1])
    log_fpr, log_tpr, t5 = roc_curve(y_final, log_y_probs_final[:, 1])

    plt.style.use("seaborn-darkgrid")
    plt.rcParams["patch.force_edgecolor"] = True
    plt.figure(figsize=(12, 6))
    plt.plot(rf_fpr, rf_tpr, label='Random Forest', alpha=0.5)
    plt.plot(ab_fpr, ab_tpr, label='Ada Boosting', alpha=0.5)
    plt.plot(gdbr_fpr, gdbr_tpr, label='Gradiant Boosting', alpha=0.5)
    plt.plot(dt_fpr, dt_tpr, label='Decision Tree', alpha=0.5)
    plt.plot(log_fpr, log_tpr, label='Logistic Reg', alpha=0.5)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity, Recall)")
    plt.title("ROC plot")
    plt.legend()
    plt.savefig("ROC_Curve.png")
    plt.show()
