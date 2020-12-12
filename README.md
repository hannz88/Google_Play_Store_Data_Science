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
- [Data visualisation and analysis](#data-visualisation-and-analysis)
- [Statistical analysis](#statistical-analysis)
- [Sentiment analysis](#sentiment-analysis)
- [Insights](#Insights)
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

## Data visualisation
One of basics before running analysis is to analyse the correlations between the variables. The following is a correlation matrix between the variables:

<p align="center">
    <img src="https://github.com/hannz88/Google_Play_Store_Data_Science/blob/main/Graphs/correlation_matrix.png" alt="Correlation matrix of the variables" height="70%" width="70%">
</p>

It's shown that "Installs" and "Reviews" have medium, positive correlation with each other. When I applied Pearson's r, it appears that they are significant with p-value < 0. Everything else had virtually weak to no correlation.

Next, I used data visualisations to answer some of the questions and trends.

### What's the market breakdown?
<p align="center">
    <img src="https://github.com/hannz88/Google_Play_Store_Data_Science/blob/main/Graphs/market_breakdown.png" alt="Breakdown of the market">
</p>

From the pie chart, we could see that Game apps has a large part in the market. This is then followed by Communication and Tools.

### What is the distribution of ratings across different category?
<p align="center">
    <img src="https://github.com/hannz88/Google_Play_Store_Data_Science/blob/main/Graphs/violinplot.png" alt="Violin plot of ratings across category">
</p>

From the violin plot, a few things could be seen:

- Apps in the categories of Business, Dating, Medical and Tools have a large variance in ratings.
- More than 50% of Apps in Art and Design, Books and Reference, Education, Events and Tools have higher than average ratings.
- More than 50% of Apps in Dating have lower than average ratings.

### What is the size strategy?
<p align="center">
    <img src="https://github.com/hannz88/Google_Play_Store_Data_Science/blob/main/Graphs/size_strategy.png" alt="Difference in size strategy">
</p>
<p align="center">
    <img src="https://github.com/hannz88/Google_Play_Store_Data_Science/blob/main/Graphs/size_category.png" alt="Size for each category">
</p>

From the scatter-histogram, it becomes evident that a large number of the apps are less than 20 Mb and more than 50% of them have the ratings of 4 and above. It appears that the apps tend to keep themselves to the light weight rather than being bulky. From the scatter plot, it appears that apps from Game, Family and/or Medical tend to be pretty wide-spread in regards to size. However, these apps also seem to be pretty well-received as they have ratings of 3.5 and above. Parenting, Tools and Video players app appear to be smaller in size as they are mostly 40Mb and below but in terms of ratings, it appears that they tend to be between the range of 3.0 to 4.5.

### What is the pricing strategy?
<p align="center">
    <img src="https://github.com/hannz88/Google_Play_Store_Data_Science/blob/main/Graphs/Pricing_apps.png" alt="Difference in Price strategy" height="50%" width="50%">
</p>
<p align="center">
    <img src="https://github.com/hannz88/Google_Play_Store_Data_Science/blob/main/Graphs/price_each_category.png" alt="Price for each category">
</p>

From the pie chart, only 7.8% of the Apps are in the Paid category. Out of the paid category, more than 50% of them are $100 and below. Somebody expressed their suprise that Game apps are less than $100. A gamer would understand that the companies do not earn their revenue through the sales of the app but the in-app purchases. Furthermore, only 10 apps are above $100. Let's take a look at them.

<p align="center">
    <img src="https://github.com/hannz88/Google_Play_Store_Data_Science/blob/main/Graphs/I_am_rich.png" alt="I am rich apps">
</p>

When looking at those expensive apps, I am legit shooketh. Smh. Wikipedia claimed that the "I am rich" apps were apparently "a work of art with no hidden function at all" and their creation was for no other reason than to show off that they could afford it. In other words, they are just flexing that they're rich. Why tho??

<p align="center">
    <img src="https://github.com/hannz88/Google_Play_Store_Data_Science/blob/main/Graphs/rich.gif" alt="Make it rain gif">
</p>

## Statistical analysis
Given the data, there are two questions that I'm curious about and decided to test them. 

1) Is there a difference in popularity between Free and Paid apps?
2) Is there a difference in ratings between the different categories?

### Is there a difference in popularity between Free and Paid apps?
To answer this question, I used "Installs" as a measurement. The reason is simply that if an app is popular it is more likely to get spread by word-of-mouth. First, let's do a quick exploratory analysis of the difference between Free and Paid

<p align="center">
    <img src="https://github.com/hannz88/Google_Play_Store_Data_Science/blob/main/Graphs/boxplot_free_paid.png" alt="Boxplot to compare popularity between free and paid">
</p>

Given that the boxplot of the number of downloads appear to have some overlap, it might still have a significant difference between the different types. We could use a variant of t-test to test the differences. I performed **levene's test** and **shapiro-wilk's test** for homogeneity of variance and normality respectively. The results showed that both assumptions are violated (both p-values < 0.05).

```
# levene's test
from scipy.stats import levene
x = store_df.loc[store_df["Type"]=="Free", "Installs_log"]
y = store_df.loc[store_df["Type"]=="Paid", "Installs_log"]
s, p = levene(x,y)
p
>>> 1.3113456031787633e-20

# shapiro-wilk's test
from scipy.stats import shapiro
s,p = stats.shapiro(x)
>>> 4.021726592612225e-43
```

The homogeneity of variance and normality are violated, so **student t-test** is not advisable. So, an unpaired, non-parametric test should be used. Under these conditions, **Mann whitney test** is probably the most appropriate. In general, Mann Whitney's assumptions are:

- observations from both groups are independent from each other
- responses are at least ordinal (ie, you can say which is higher)

Since the assumptions are met, we'll go ahead and use the test.

```
# Mann-Whitney
from scipy.stats import mannwhitneyu
stats.mannwhitneyu(x,y)
>>> MannwhitneyuResult(statistic=1685312.5, pvalue=1.2531215783547303e-116)
```

We can reject the null hypothesis that the sample distributions are equal between the groups (p-value < 0.05, U= 1685312.5)

### Is there a difference in ratings between the different categories?

Given that there are multiple levels (aka multiple categories within an independent variable), I decided to use **One-way ANOVA** at first. However, the residuals did not meet the assumption of normality as visible from the QQ plot of the residuals.  

<p align="center">
    <img src="https://github.com/hannz88/Google_Play_Store_Data_Science/blob/main/Graphs/qqplot_of_residuals.png" alt="Residuals QQ plot">
</p>

Then, I tried log transformation but it did not help either. Therefore, I decided to use non-parametric test, specifically **Kruskal-Wallis test**.  Before we conduct Kruskal-Wallis test, there are a few assumptions that are needed to be met:

1) Samples drawn are random 
2) Observations are independent

Both of these assumptions are met because each app is a unique entry so they are independent of each other. Note: Scipy does not have a function that will give you the effect size of Kruskal-Wallis test but it's easy to obtain it using the s-value from the test. 

```
# Kruskal-Wallis test
from scipy import stats
s, p = stats.kruskal(*[group["Ratings_imp"].values for name, group in store_df.groupby("Category")])
>>> 291.9695989365334 9.87269222844556e-44

# Effect size
def kruskal_effect_size(h, n, k):
    """
    Return the effect size of Kruskal-Wallis test.
    H = H-value of statistics of Kruskal-Wallis
    n = number of observations
    k = number of groups
    The formulas is from Tomczak and Tomczak (2014)
    """
    return h * (n+1)/(n**2 - 1)
n = len(store_df)   
k = len(store_df["Category"].unique())
kruskal_effect_size(h = s, n=n, k=k)
>>> 0.03022772532731477
```
Kruskal-Wallis test showed that there is a significant difference among the ratings of different categories (p-value < 0, H-value = 291.97) but the effect is weak (eta-squared = 0.03).

## Sentiment analysis
The data also comes with a dataset documenting the reviews of users for some apps. We'll go ahead and pull out the most common words for Free vs Paid apps.

<img src="https://github.com/hannz88/Google_Play_Store_Data_Science/blob/main/Graphs/free_word_cloud.png" width="400"/> <img src="https://github.com/hannz88/Google_Play_Store_Data_Science/blob/main/Graphs/paid_word_cloud.png" width="400"/> 
<p align="center">
    <img src="https://github.com/hannz88/Google_Play_Store_Data_Science/blob/main/Graphs/sentiment_polarity_distribution.png" alt="Distribution of Sentiment Polarity">
</p>

Positive words associated with Free Apps are **free**, **love**, **great**, and **good** while positive words associated with Paid Apps are **great**, **easy**, and **cute**.

Negative words associated with Free Apps are **problem**, **bad**, and **fix** while negative words associated with Paid Apps are **issue** and **problem**.

In terms of sentiment polarity, there doesn't seem to be a difference between Free and Paid apps as there are a lot of overlap. However, there seems to be more negative reviews for Free Apps which are marked as outliers.

## Insights
In short, the current findings are:
- users seem to tend to prefer light-weight app, judging by the higher ratings
- Top-rated apps tend to be less than 40 Mb
- Having said that, this doesn't seem to apply to Game apps
- Free apps are more popular than Paid apps (that's a no-brainer, tbh; still, I proved it with stats!)
- There is a difference in ratings across the categories, albeit a small one.
- Users seem to merciless when reviewing Free Apps

## Machine Learning
In order to keep this page short (well, I tried), I've split the machine learning part into a separate [page](https://github.com/hannz88/Google_Play_Store_Data_Science/blob/main/Machine_learning.md)
