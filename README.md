# A-B-Testing-Red-and-White-Wine
## Introduction 
In this case study, red and white wine quality was analayzed from their respective data sets. The wine samples are from the north of Portugal.

The data sets contain a quality rating, from a scale of 1 to 10, for each individual sample. 

## Data Set 

### Source 
  >The two data sets are related to **red** and **white** variants of the Portuguese 'Vinho Verde' wine. Due to 
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

This data set has been traditionally used for classification and regression modeling. However,the goal of this project is to apply A/B Hypothesis Testing and provide business oriented recommendations based on the result. 

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
  >The output is shown below
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

#### Initial Remarks 
A quick inspection reveals a similar mean in quality score between the two wines. However, the distrbutions for the white wine appears to be normal while the red wine quality score is close to normal but slightly narrower with a slight left skew. We do not perform any data transformations but we do check for outliers below in the form of a box-and-whisker plot.

```python
# Check for outliers
# Box and Whisker plot 
fig, axes = plt.subplots(1,2)
axes[0].boxplot(red_wine_df['quality'])
axes[0].set_xlabel('Red Wine')
axes[0].set_xticks([])

axes[1].boxplot(white_wine_df['quality'])
axes[1].set_xlabel('White Wine')
axes[1].set_xticks([])

#Labels and title 
for ax in axes:
    ax.set_ylabel("Quality Score")
plt.suptitle("Box-and-Whisker Plot of Wine Quality")
plt.show()
```
 >The output is shown below

![wine box and whisker plot](https://github.com/user-attachments/assets/2094298a-85c2-461c-ad81-40bf498e187d)

There are outliers in both data sets. If we were confident that both data sets were normally distributed we might opt to remove outliers past two standard deviations in both directions from the mean. However, we take a more pragmatic approach and remove ouliers via the _IQR Method_. The IQR method removes data _1.5_ times the interquartile range in both directions from the data set. 

```python
#There are outliers, use the IQR method
Q1_red=red_wine_df['quality'].quantile(.25)
Q1_white=white_wine_df['quality'].quantile(.25)

Q3_red=red_wine_df['quality'].quantile(.75)
Q3_white=white_wine_df['quality'].quantile(.75)

IQR_red = Q3_red-Q1_red
IQR_white = Q3_white-Q1_white

lower_red=Q1_red-1.5*IQR_red
lower_white=Q1_white-1.5*IQR_white

upper_red=Q3_red+1.5*IQR_red
upper_white=Q3_white+1.5*IQR_white

#Create an array of boolean value indicating the outlier rows
lower_array_red=np.where(red_wine_df['quality'] <= lower_red)[0]
lower_array_white=np.where(white_wine_df['quality'] <= lower_white)[0]
upper_array_red=np.where(red_wine_df['quality'] >= upper_red)[0]
upper_array_white=np.where(white_wine_df['quality'] >= upper_white)[0]

#removing the outliers 
red_wine_df.drop(index=lower_array_red,inplace = True)
red_wine_df.drop(index=upper_array_red,inplace = True)
white_wine_df.drop(index = lower_array_white,inplace = True)
white_wine_df.drop(index= upper_array_white, inplace = True)
```
We quickly check the descriptive statitics of the quality column of the red and white wine data sets after removing the outliers. We also redraw the box-and-whisker plot as a double check. 

```python
#Quickly Check the descriptive statistics of red and white wine
print(red_wine_df['quality'].describe())
print(white_wine_df['quality'].describe())
#Print box and whisker plot without outliers 

fig, axes = plt.subplots(1,2)
axes[0].boxplot(red_wine_df['quality'])
axes[0].set_xlabel('Red Wine')
axes[0].set_xticks([])

axes[1].boxplot(white_wine_df['quality'])
axes[1].set_xlabel('White Wine')
axes[1].set_xticks([])

#Labels and title 
for ax in axes:
    ax.set_ylabel("Quality Score")
plt.suptitle("Box-and-Whisker Plot of Wine Quality")
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
                    hspace=0.4, wspace=0.3)
plt.show()
```
  >output is shown below
```
count    1571.000000
mean        5.625716
std         0.745227
min         4.000000
25%         5.000000
50%         6.000000
75%         6.000000
max         7.000000
Name: quality, dtype: float64
count    4698.000000
mean        5.807791
std         0.774217
min         4.000000
25%         5.000000
50%         6.000000
75%         6.000000
max         7.000000
Name: quality, dtype: float64
```
![box and whisker with outliers removed](https://github.com/user-attachments/assets/f1682ca0-65cf-4782-9a01-6af32e2838f1)

Visually we can verify that the outliers have been removed. The red wine data set had 28 samples removed and the white wine data set had 200 samples removed. The mean's of both red and white wine decreased slightly. Now that the outliers have been removed we proceed to the next step of hypothesis testing. 

## Hypothesis Testing 
### Designing the Expirement 
We want to test if their exist a difference in the mean quality score between red and white wine. Even though white wine quality score has a slightly larger average quality score we cannot say for certain that signicantly better than red wine. Therefore, without making any assumptions, we will perform a **two-tailed test**:  
<p align ="center">
  $$H_0: \mu_{\text{red}} = \mu_{\text{white}} $$
</p>
<p align ="center">
  $$H_A: \mu_{\text{red}} \ne \mu_{\text{white}} $$
</p>

### Variable Selection and Modification 
Traditionally there are two groups, a _control_ and a _treatment_, however in this case study assigning red and white wine to either group is arbitrary. For our purposes we will treat the red wine as the control and the white wine as the treatment. We will codify red and white wine as binary variables later on. 
  > - 0: Red Wine
  > - 1: White Wine
### Verifying Sample Size 
We are are only testing the wine samples from the north of Portugal. However, for verification purposes we complete _power analysis_ to verify that out sample size is sufficient. Before we do that we have to make certain assumptions before calculating our _power analysis_. That an average score for wine quality is 5 and anything above 5 is considered _above average_ in quality. Lets assume that currently 50% of the red wine is above average and intervention takes place if 55% of white wine is above average. 

  -**Power of the test** $$(1-\beta)$$ We use standard convention and set beta to .2.  
  -**Alpha** $$(\alpha)$$ We use standard convention and set alpha to 0.05  
  -**Effect Size** Discussed above, we expect there to be a 5% difference in quality between the two wines  
The code snippet below calculates required sample size
```python
##Calculate the minimum number of observations for each group
import statsmodels.stats.api as sms
from math import ceil
effect_size= sms.proportion_effectsize(0.5,0.55)
required_size=sms.NormalIndPower().solve_power(
    effect_size,
    power=0.8,
    alpha=0.05,
    ratio=1
)
required_size = ceil(required_size)
print(required_size)
```
  >output is shown below
```
1565
```
Based on our calculation, we require a minimum of 1565 observations from both red and white wine catergories. Fortunately, our data set meets this requirement.
### Sampling
Data processing occured above, so at this stage we take a sample size of 1565 for both red and white wine. 
```python
#Take a random sample of 1565 from both groups 
sample_red = red_wine_df.sample(n=1565)
sample_white=white_wine_df.sample(n=1565)
frames=[sample_red,sample_white]
#Combined sample will be used later 
combined_sample = pd.concat(frames)
```
### Normality Assumption (Shapiro-Wilk Test)
The Shapiro-Wilk Test is a commonly used statistical tool to conduct a normality check, in particular, we are testing if the data in our sample is normally distributed. 
<p align="center"> 
The Shapiro-Wilk test statistic $W$ is defined as:

$$\begin{equation}
W = \frac{\left( \sum_{i=1}^{n} a_i x_{(i)} \right)^2}{\sum_{i=1}^{n} (x_i - \bar{x})^2}
\end{equation}$$

where:
$$\begin{itemize}
    \item $x_{(i)}$ is the $i$-th order statistic (i.e., the $i$-th smallest value in the sample),
    \item $\bar{x}$ is the sample mean: $\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$,
    \item $a_i$ are constants computed from the means, variances, and covariances of the order statistics of a sample of size $n$ from a standard normal distribution,
    \item $n$ is the sample size.
\end{itemize}$$

The numerator can also be written as:
$$\begin{equation}
\left( \sum_{i=1}^{n} a_i x_{(i)} \right)^2 = (\bm{a}^T \bm{x}_{(sorted)})^2
\end{equation}$$

The denominator is the sample variance (times $n$), ensuring that $W$ lies in the interval $(0, 1]$.

 A $W$ value close to 1 indicates normality, while smaller values suggest departure from normality. The test's significance is assessed using precomputed critical values or p-values based on $W$ and $n$.
</p>
