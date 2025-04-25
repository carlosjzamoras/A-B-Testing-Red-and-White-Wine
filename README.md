# A-B-Testing-Red-and-White-Wine
## Introduction 
In this case study, red and white wine quality was analayzed from their respective data sets. The wine samples are from the north of Portugal.

The data sets contain a quality rating, from a scale of 1 to 10, for each individual sample. 

## Data Set 

### Source 
  >The two data sets are related to **red and ** white variants of the Portoguese 'Vinho Verde' wine. Due to 
  >privacy and logistic issues, only physiochemical (inputs) and sensory (the output) variables are  
  >available (e.g. there is no data about grape types, wine brands, wine selling price, etc.).

[Reference Paper](https://archive.ics.uci.edu/dataset/186/wine+quality)

### Variable Information

Variables:
  * fixed acidity
  * volatile acidity
  * citric acid
  * residual sugar 
  * chlorides
  * free sulfur dioxide
  * total sulfur dioxide
  * density
  * pH
  * sulphates
  * alcohol
  * quality (score between 0 and 10)

## Problem Statement
Our client is concerned with  assessing the quality of red and white wine. Specifically, determining if there is a significant difference in the quality between the two types of wine. The findings could lead to changes in material orders, wine selection availability to customers, and potential changes in the cultivating procress. 

This data sent has been traditionally used for classification and regression modeling. The goal of this project is to apply A/B Hypothesis Testing and provide business oriented recommendations based on this result. 

## Data Proceccessing
The first step in data processing is importing the revelant libraries and loading the csv. 
```python
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
red_wine_df = pd.read_csv('/Users/carloszamora/Downloads/wine+quality/winequality-red.csv',sep=";")
white_wine_df = pd.read_csv('/Users/carloszamora/Downloads/wine+quality/winequality-white.csv',sep=";")
# Create additional column in both dataframes specifying the color 
red_wine_df['Color'] = 'red'
white_wine_df['Color'] = 'white'
#Combine the data frames 
frames = [red_wine_df,white_wine_df]
combined_wine = pd.concat(frames)
```
We examine the summary statistics of the quality score for each wine type. Additionally, we take a look at a look at the distribution of quality for score for each type of wine. This will provide high-level insight on how to performing our hypothesis test. 

```python
print(red_wine_df['quality'].describe())
print(white_wine_df['quality'].describe())
#Take a Look at the distribution of the wine quality based on color

```
The output is shown below
```
count    1599.000000
mean        5.636023
std         0.807569
min         3.000000
25%         5.000000
50%         6.000000
75%         6.000000
max         8.000000
Name: quality, dtype: float64
count    4898.000000
mean        5.877909
std         0.885639
min         3.000000
25%         5.000000
50%         6.000000
75%         6.000000
max         9.000000
Name: quality, dtype: float64
```
![red_white_wine_distribution](https://github.com/user-attachments/assets/a7f7bc27-da1c-4356-8309-e7ecc1055c32)

