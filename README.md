# Predicting Churn at Rideshare Company

_7/13/18_
_Mike Irvine, Pat Hottovy, Tim Marlowe_

<p align="center">
  <img width="460" height="300" src="https://github.com/mikeirvine/dsi-ml-case-study/blob/master/images/uber.jpg">
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
  <img width="600" height="400"
src="https://github.com/phottovy/dsi-ml-case-study/blob/master/images/decision_tree.png">
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

||Accuracy|ROC AUC|
|---|---|---|
|Logistic Regression|.884|.852|
|Decision Tree|.883|.859|
|Random Forest|.891|.863|
|Gradient Boosting|.887|.863|
|Adaptive Boosting|.885|.848|

__Final Holdout Data__

||Accuracy|ROC AUC|
|---|---|---|
|Logistic Regression|.883|.850|
|Decision Tree|.898|.874|
|Random Forest|.899|.871|
|Gradient Boosting|.926|.907|
|Adaptive Boosting|.845|.883|

<p align="center">
  <img width="800" height="600"
src="https://github.com/phottovy/dsi-ml-case-study/blob/master/images/ROC_Curve.png">
</p>

As Random forest and gradient boosting modeling produced the highest levels of accuracy and roc_auc scores for our training data, we would likely recommend using one of these two models to predict churn.

## Feature importance

Below is the feature importance for our Gradient boosting model (top 5):

|Feature| Importance|
|---|---|
|ride_last_two_weeks| 0.32|
|avg_dist| 0.212|
|weekday_pct| 0.099|
|avg_rating_of_driver| 0.09|
|trips_in_first_30_days| 0.065|

It seems clear that having ridden in the last two weeks is the most important feature. We'd suggest a campaign to re-engage people who have not ridden in a few weeks - perhaps a coupon for money off of a ride.

## Code
The following code was used to generate these conclusions:
* [rideshare_casestudy.py](https://github.com/phottovy/dsi-ml-case-study/blob/master/src/rideshare_casestudy.py)
