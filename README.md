# A Data Science Project on Google Play Store Data
[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)

![Uses Numpy](https://img.shields.io/badge/uses-numpy-yellow)
![Uses Pandas](https://img.shields.io/badge/uses-Pandas-green)
![Uses Scikit-learn](https://img.shields.io/badge/uses-scikit--learn-blue)

## Introduction
This a side-project using 2018 Google Play Store data available on Kaggle. The original data came with two datasets: one on the app data and one on customer reviews for some of the apps. The main skills used/ shown in this project are: 

- Data cleaning/ wrangling including imputation
- Exploratory data analysis
- Data visualisation
- Statistical analysis
- Machine learning

## Table of content

- [Data cleaning](#data-cleaning)
- [Data visualisation](#data-visualisation)
- [Statistical analysis](#statistical-analysis)
- [Machine learning](#machine-learning)

## Data cleaning
The app dataset contains 10841 rows with 13 columns. However, some rows are duplicated and when these are removed, we're left with 9660 rows. There are a few categories that are of interest to us, namely:

- `Category`: Nominal data. There are 33 categories, such as Game, Beauty, Comics, Finance, etc.
- `Ratings`: Ratio data. An average of ratings from all the reviews for the app.
- `Type`: Nominal data. Whether the app is Free or Paid
- `Installs`: Ratio data. The number of downloads.
- `Price`: Technically a numerical data with true zero which makes it ratio but in the dataset, it acts more like an interval data. The price of the app.

On top of that, some of values are missing, which makes it difficult to perform downstream analysis and machine learning. For a few of the entries, the missing values (mv) were due to web-scrapping error and it was easy to fix by replacing the mv with the data available online. For a lot of them however, it was too much to gather information separately especially when Google Play Store does not provide API access. In order to avoid massive loss of information, I decided to impute some of the missing values.  A lot of the variables have skewed distribution. I decided not to use mean replacement which was commonly used as I felt that this would distort the relationship between the variables. Instead, I'm going to impute using a method from sklearn that mimics missForest in R. We're using IterativeImputer with ExtraTreesRegressor to mimic missForest in R. missForest imputes mising data using mean/mode first then for each variable with mv, it fits random forest and imputes the missing part. It does not assumes normality which is great as the distribution is obviously anything but, as can be seen on the exploratory visualisation below.

<p align="center">
    <img src="https://github.com/hannz88/Google_Play_Store_Data_Science/blob/main/Graphs/EDA.png" alt="Exploratory Analysis of the variables of interest">
</p>