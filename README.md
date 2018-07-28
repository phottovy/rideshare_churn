# Predicting Churn at Rideshare Company

_7/13/18_
_Mike Irvine, Pat Hottovy, Tim Marlowe_

<p align="center">
  <img src="https://github.com/phottovy/rideshare_churn/blob/master/images/uber.jpg">
</p>

## Problem Statement
Getting new passengers is a difficult task in today's competitive ride share world. Therefore, ride share companies must work to retain passengers. Our modeling task today was to predict which passengers would leave the company (churn), as defined by taking no rides in the last 30 days.

## Data Preparation and EDA
The dataset provided was in the form of one passenger per row, and had the following features:

  * __avg_dist__: Average distance traveled in rides by passenger
  * __avg_rating_by_driver__: Average rating given to the passenger by drivers
  * __avg_rating_of_driver__: Average rating given to drivers by the passenger
  * __avg_surge__: Average surge pricing at the time of ride
  * __city__: What city the passenger resides in (waddup GOT)
  * __last_trip_date__: When the passenger last traveled with the ride company
  * __phone__: Phone type - iphone or android
  * __signup_date__: When the passenger signed up for the service
  * __surge_pct__: Percent of time passenger rode during surge
  * __trips_in_first_30_days__: Number of times passenger traveled in the first 30 days
  * __luxury_car__: Whether the passenger used a luxury car with the service
  * __weekday_pct__: Percent of rides taken on weekday

Using these categories, we needed to calculate our target (churn), ensure we had dealt with data leakage, and impute data for a single feature with missing data.

#### Calculating our target and addressing data leakage through feature engineering
In order to calculate churn, we used the _last_trip_date_ feature. If a passenger had not taken a ride within thirty days of the date of extract, we defined them as having left or "churned". For the training data set, the proportions were as follows:

|"Churned"|Persisting|
|---|---|
|62.3%|37.7%|

This indicates that we did not have a large problem with class imbalance.

Having calculated churn, the risk was that we would encounter data leakage by using _last_trip_date_ as a feature, as any passenger who had not traveled for 30 days would be perfectly predicted as churning (information we would not have in real time). To mimic real-time decision-making, we created a feature called _ride_last_two_weeks_ indicating whether the passenger had ridden in the last two weeks and dropped _last_trip_date_ from our features.

The following code executed both the calculation of churn for passengers and dealt with leakage:

```python
def load_churn_data(filename):
    df = pd.read_csv(filename)
    df.dropna(subset=['avg_rating_by_driver'],inplace=True)
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'], yearfirst=True, format='%Y/%m/%d')
    df['signup_date'] = pd.to_datetime(df['signup_date'], yearfirst=True, format='%Y/%m/%d')
    churn_date = pd.datetime(2014,7,1) - pd.to_timedelta(30, unit='D')
    two_week_date = pd.datetime(2014,7,1) - pd.to_timedelta(14, unit='D')
    df['ride_last_two_weeks'] = df["last_trip_date"] >= two_week_date
    df['churn'] = df["last_trip_date"] < churn_date
    return df
  ```

#### Data Imputation for avg_rating_of_driver

Roughly 7000 out of 40,000 training riders never rated their drivers. They took, on average, fewer rides, than their counterparts who did rate their drivers, but given that they were a large part of the data set, we did not want to drop them.

We used K-nearest neighbors regression to impute driver rating values, using a gridsearch to do so. Because of the curse of dimensionality, the R2 of our imputation, even for the best model identified (k = square root(n)) was extremely low (.01)

## Modeling

For our models, we tried the following five classifiers:
* Logistic Regression
* Single Decision Tree
* Random Forest
* Gradient Boosting Classifier
* Adaptive Boosting Classifier

Below is an image of a basic decision tree for this classification problem, with a depth of 4 splits. This type of decision tree is the basis for our Decision Tree, Random Forest, Gradient Boosting, and Adaptive Boosting modeling:
<p align="center">
  <img
src="https://github.com/phottovy/rideshare_churn/blob/master/images/decision_tree.png">
</p>

We utilized Gridsearch to identify the optimal parameters for each model, using the following code (as applied to random forests below):

```python
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
```

Using this code, we then identified the best parameters and applied them to the entire training dataset. The accuracy scores from each model on training and test data were :

__Training Data__

||Accuracy|Precision|Recall|F1-Score|ROC AUC|
|---|--:|--:|--:|--:|--:|
|Logistic Regression|0.89|0.86|0.98|0.92|0.86|
|Decision Tree|0.88|0.87|0.96|0.91|0.86|
|Random Forest|0.90|0.86|0.98|0.92|0.86|
|Gradient Boosting|0.89|0.87|0.96|0.91|0.86|
|Adaptive Boosting|0.89|0.85|1.00|0.92|0.85|

__Final Holdout Data__

||Accuracy|ROC AUC|Precision|Recall|F1-Score|
|---|--:|--:|--:|--:|--:|
|Logistic Regression||||||
|Decision Tree|||
|Random Forest|||
|Gradient Boosting|||
|Adaptive Boosting|||





## ROC Plot

<p align="center">
  <img
src="https://github.com/phottovy/rideshare_churn/blob/master/images/ROC_Curve.png">
</p>

As Random forest and gradient boosting modeling produced the highest levels of accuracy and roc_auc scores for our training data, we would likely recommend using one of these two models to predict churn.

## Feature importance

Below are the feature importances for our the different models:

<p align="center">
  <img
src="https://github.com/phottovy/rideshare_churn/blob/master/images/random_forest_feat_importance.png">
</p>

Random Forest:

|    | feature                |   importance |
|---:|:-----------------------|-------------:|
|  1 | ride_last_two_weeks    |      0.62499 |
|  2 | avg_rating_by_driver   |      0.06696 |
|  3 | surge_pct              |      0.05664 |
|  4 | King's Landing         |      0.05292 |
|  5 | weekday_pct            |      0.03764 |
|  6 | avg_surge              |      0.03496 |
|  7 | trips_in_first_30_days |      0.02428 |
|  8 | luxury_car_user        |      0.02341 |
|  9 | avg_dist               |      0.01883 |
| 10 | iPhone                 |      0.01704 |
| 11 | Android                |      0.01578 |
| 12 | avg_rating_of_driver   |      0.01201 |
| 13 | Astapor                |      0.00892 |
| 14 | Winterfell             |      0.00564 |

<p align="center">
  <img
src="https://github.com/phottovy/rideshare_churn/blob/master/images/adaptive_boosting_feat_importance.png">
</p>

Adaptive Boosting:

|    | feature                |   importance |
|---:|:-----------------------|-------------:|
|  1 | weekday_pct            |         0.26 |
|  2 | King's Landing         |         0.2  |
|  3 | avg_rating_by_driver   |         0.13 |
|  4 | luxury_car_user        |         0.12 |
|  5 | iPhone                 |         0.1  |
|  6 | surge_pct              |         0.05 |
|  7 | ride_last_two_weeks    |         0.05 |
|  8 | Android                |         0.05 |
|  9 | Astapor                |         0.04 |
| 10 | avg_dist               |         0    |
| 11 | avg_rating_of_driver   |         0    |
| 12 | avg_surge              |         0    |
| 13 | trips_in_first_30_days |         0    |
| 14 | Winterfell             |         0    |

<p align="center">
  <img
src="https://github.com/phottovy/rideshare_churn/blob/master/images/gradiant_boosting_feat_importance">
</p>

Gradiant Boosting:

|    | feature                |   importance |
|---:|:-----------------------|-------------:|
|  1 | ride_last_two_weeks    |      0.32084 |
|  2 | avg_dist               |      0.2125  |
|  3 | weekday_pct            |      0.09925 |
|  4 | avg_rating_of_driver   |      0.09056 |
|  5 | trips_in_first_30_days |      0.0649  |
|  6 | avg_rating_by_driver   |      0.05444 |
|  7 | surge_pct              |      0.0532  |
|  8 | avg_surge              |      0.0406  |
|  9 | luxury_car_user        |      0.01701 |
| 10 | King's Landing         |      0.01691 |
| 11 | iPhone                 |      0.0089  |
| 12 | Astapor                |      0.00771 |
| 13 | Winterfell             |      0.00679 |
| 14 | Android                |      0.00639 |

<p align="center">
  <img
src="https://github.com/phottovy/rideshare_churn/blob/master/images/decision_tree_feat_importance.png">
</p>

Decision Tree:

|    | feature                |   importance |
|---:|:-----------------------|-------------:|
|  1 | ride_last_two_weeks    |      0.81593 |
|  2 | avg_rating_by_driver   |      0.03231 |
|  3 | King's Landing         |      0.02968 |
|  4 | weekday_pct            |      0.02422 |
|  5 | avg_dist               |      0.01919 |
|  6 | luxury_car_user        |      0.01792 |
|  7 | trips_in_first_30_days |      0.01204 |
|  8 | surge_pct              |      0.01201 |
|  9 | avg_rating_of_driver   |      0.01071 |
| 10 | Android                |      0.0101  |
| 11 | avg_surge              |      0.00635 |
| 12 | iPhone                 |      0.0056  |
| 13 | Winterfell             |      0.00262 |
| 14 | Astapor                |      0.00132 |

<p align="center">
  <img
src="https://github.com/phottovy/rideshare_churn/blob/master/images/logistic_regression_feat_importance.png">
</p>

Logistic Regression:

|    | feature                |     coef |
|---:|:-----------------------|---------:|
|  0 | ride_last_two_weeks    | -4.23607 |
|  1 | King's Landing         | -0.41918 |
|  2 | luxury_car_user        | -0.40778 |
|  3 | iPhone                 | -0.3195  |
|  4 | Astapor                |  0.29361 |
|  5 | trips_in_first_30_days | -0.23531 |
|  6 | avg_dist               |  0.15473 |
|  7 | Android                |  0.12595 |
|  8 | surge_pct              | -0.11914 |
|  9 | avg_surge              |  0.11285 |
| 10 | weekday_pct            |  0.07656 |
| 11 | avg_rating_by_driver   |  0.06791 |
| 12 | Winterfell             |  0.06107 |
| 13 | avg_rating_of_driver   |  0.03122 |


It seems clear that having ridden in the last two weeks is the most important feature. We'd suggest a campaign to re-engage people who have not ridden in a few weeks - perhaps a coupon for money off of a ride.

## Code
The following code was used to generate these conclusions:
* [rideshare_casestudy.py](https://github.com/phottovy/rideshare_churn/blob/master/src/rideshare_casestudy.py)
