# Machine Learning To Predict Ratings for Google Play Store App

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)

![Open to suggestions](https://img.shields.io/badge/open%20to-suggestions-yellow)
![Uses Scikit-learn](https://img.shields.io/badge/uses-scikit--learn-blue)

This is a continuation of the Data Science project using the Google Play Store data. The skills used and shown in this project are:

- Feature selection
- Model training
- Hyperparameter tuning

## Table of Content

- [Recap](#recap)
- [Feature selection](#feature-selection)
- [Model selection](#model-selection)
- [RandomForestRegressor with Dummy Variables](#RandomForestRegressor-with-dummy-variables)
- [Conclusion](#conclusion)

## Recap
The data was scrapped from 2018 Google Play Store. There were missing values as well as incorrect entries originally. I corrected the incorrect entries and imputed the missing values using `ExtraTreeRegressor` to mimic `miss Forest`, a commonly used package in R. Click [here](https://github.com/hannz88/Google_Play_Store_Data_Science/blob/main/README.md) to see how I cleaned them in details and [here](https://github.com/hannz88/Google_Play_Store_Data_Science/blob/main/Android%20app%20sales.ipynb) for the graphical outputs and statistical analysis.

Again, the original variables of interest are:

- `Category`: Nominal data. There are 33 categories, such as Game, Beauty, Comics, Finance, etc.
- `Ratings`: Ratio data. An average of ratings from all the reviews for the app.
- `Type`: Nominal data. Whether the app is Free or Paid
- `Installs`: Ratio data. The number of downloads.
- `Price`: Technically a numerical data with true zero which makes it ratio but in the dataset, it acts more like an interval data. The price of the app.

However, machine learning models don't normally take in categorical string data, so I've encoded the categorical columns into numerical using `LabelEncoder`. There is one caveat to the method I was using, however. I applied LabelEncoding to turn them into numerical. When training, regression will think of them as ordinal, ie 1 is better than 2. However, we have 33 levels in Category alone, which is a bit much to use one-hot encoder or dummy coding. Regardless, I experimented with dummy encoding afterwards to see if there's a difference. 

Here's an example of the dataframe I used at first:

<p align="center">
    <img src="https://github.com/hannz88/Google_Play_Store_Data_Science/blob/main/Graphs/preprocessed_df.png" alt="Preprocessed dataframe used for machine learning at first">
</p>

## Feature selection
According to Jason Brownlee PhD, feature selection is the process of reducing the number of input variables used for predictive modelling. Here, I've used a few methods on the `LabelEncoder` encoded data:

- Recursive Feature Elimination (rfe) using `GradientBoostingRegressor`. For more information on rfe, please refer to the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html) in sklearn.
- SelectFromModel using `RandomForestRegressor` and `XGBoostRegressor`. This method selects features based on [feature importance](https://scikit-learn.org/dev/auto_examples/feature_selection/plot_select_from_model_diabetes.html)
- Mutual information which measures the contribution of a variable towards another variable

In order to decide which features aka independent variables to use, I aggregated the results into a table:

<p align="center">
    <img src="https://github.com/hannz88/Google_Play_Store_Data_Science/blob/main/Graphs/feature_selection.png" alt="Table of aggregated results from different feature selection methods">
</p>

From the table, we can see that `Installs_scaled` and `Category_num` were selected by 4 models while `Reviews_scaled` and `Price_scaled` were selected by 3. I dedcided to use the top 4.

## Model selection
Using the 4 variables, I tested on 4 different regression machine learning models: `RandomForestRegressor`, `XGBoostRegressor`, `KNNRegressor` and `SupportVectorRegressor`. For each of the model, I tested for R-squared, mean absolute error (mae) and mean squared error (mse). The results are:

<p align="center">
    <img src="https://github.com/hannz88/Google_Play_Store_Data_Science/blob/main/Graphs/metrics.png" alt="Metrics for all the models">
</p>

What we want is a high R2 which means the amount of variance the model could explain, and low mae and mse which indicates that the predicted values are as close to the true values as possible. From the table, RFRegressor has the best performance (R2 = 0.275, mae = 0.352, mse = 0.247) so I decided to use `RandomForestRegressor` 

### Model results
Using `RandomForestRegressor` with 4 variables (Category_num, Content_num, Reviews_scaled, and Installs_scaled), I optimised the Hyperparameters for maximum leaf nodes and estimators. The best model I found was `RandomForestRegressor` with max_leaf_nodes=260 and n_estimators=80. 

```
variables = ["Category_num", "Content_num", "Reviews_scaled", "Installs_scaled", "Ratings_imp"]
new_df = df[variables]
x = new_df.drop("Ratings_imp", axis=1)
y = new_df["Ratings_imp"]
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9, test_size=0.1, random_state=8)

rf_model = RandomForestRegressor(max_leaf_nodes=260, random_state=8, n_jobs=-1, n_estimators=80)
rf_model.fit(x_train, y_train)
rf_pred = rf_model.predict(x_test)
r2_rf = r2_score(y_test, rf_pred)
msr_rf = mean_squared_error(y_test, rf_pred)
mae_rf = mean_absolute_error(y_test, rf_pred)
print(f"r2 is {r2_rf}, mae is {mae_rf} and msr is {msr_rf}.")
>>> r2 is 0.3971318851179779, mae is 0.3204367515399854 and msr is 0.21141255078600324.
```

The final model with `LabelEncoding` could only explain 39.71% of the variance, sadly.

## RandomForestRegressor with Dummy Variables
I experimented with using dummy variables by turrning the Category into dummy variables. 

```
# dummy variable
data = pd.get_dummies(tmp_df, prefix=["Category"], columns=["Category"])
```

Then, I used the previous 3 categories chosen using the feature selection method: Content_num, Reviews_scaled, and Installs_scaled; along with the dummy variables. Using these variables, I tuned the hyperparameters again. For simplicity sake, I won't show the whole printouts.

```
leaf_nodes = np.arange(10, 1000, 50)
estimators = np.arange(10, 500, 10)
best_r2 = 0
best_mae = 100
for nodes in leaf_nodes:
    for n in estimators:
        tmp_model = RandomForestRegressor(max_leaf_nodes=nodes, random_state=8, n_jobs=-1, n_estimators = n)
        tmp_model.fit(x_train, y_train)
        tmp_pred = tmp_model.predict(x_test)
        tmp_r2 = r2_score(y_test, tmp_pred)
        tmp_mae = mean_absolute_error(y_test, tmp_pred)
        tmp_mse = mean_squared_error(y_test, tmp_pred)
        if (tmp_r2 > best_r2) and (tmp_mae < best_mae):
            best_r2 = tmp_r2
            best_mae = tmp_mae
            print(f"Leaf nodes is {nodes}, estimators is {n}, r2 is {tmp_r2}, mae is {tmp_mae} and mse is {tmp_mse}.")
>>>...
Leaf nodes is 210, estimators is 220, r2 is 0.41347056643793534, mae is 0.31950024814286787 and mse is 0.2056829356196616.
Leaf nodes is 210, estimators is 230, r2 is 0.4138838619725167, mae is 0.31943438795995305 and mse is 0.20553800199149752.
Leaf nodes is 260, estimators is 230, r2 is 0.4140740648976029, mae is 0.31787991608878896 and mse is 0.20547130202086247.
```

In the end, the best model is `RandomForestRegressor` using dummy variables for Category with max_leaf_nodes at 260 and n_estimators at 230. 

## Conclusion
It seems that the best model I could come up with is using Random Forest Regressor (max_leaf_nodes=260, n_estimators=230, random_state=8, n_jobs=-1) using the following variables:

- Content_num: The encoded Content Rating, ie "Everyone", "Teen", etc.
- Reviews_scaled: Number of reviews standardised.
- Installs_scaled: Number of installds standardised.
- The categories that have been encoded into dummy variables.

Using the variables and the variables with standardised data, the model achieved roughly 41% accuracy with 0.21 mean absolute error and root mean-squared error of 0.45. It's not the most ideal but it's the best I can achieve at the moment and while it is likely that there are better models, I think there's a possibility that the variables here can't completely predict the average ratings of an app. For example, if you look at the correlation matrix (from the data analysis), ratings are not strongly correlated with any variables. I understand that the dummy variables are a lot (33 in total!), however, it gave me overall better, albeit slightly, metrics. Let me know what you think, but be nice!
