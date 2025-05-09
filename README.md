# A/B Testing Red and White Wine
## Introduction 
In this case study, red and white wine quality was analayzed from their respective data sets. The wine samples are from the north of Portugal.

The data sets contain a quality rating, from a scale of 1 to 10, for each individual sample. 

## Data Set 

### Source 
  >The two data sets are related to **red** and **white** variants of the Portuguese 'Vinho Verde' wine. > 
  >Due to privacy and logistic issues, only physiochemical (inputs) and sensory (the output) variables are  
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
Our client is concerned with  assessing the quality of red and white wine. Specifically, determining if there is a significant difference in the quality between the two types of wine. The findings could lead to changes in material orders, wine selection availability to customers, and potential changes in the cultivating process. 

This data set has been traditionally used for classification and regression modeling. However, the goal of this project is to apply A/B Hypothesis Testing and provide business oriented recommendations based on the result. 

## Data Proceccessing
The first step in to import the revelant libraries and loading the csv. 
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
We examine the summary statistics of the quality score for each wine type. Additionally, we take a look at  the quality score distribution each type of wine. This will provide high-level insight on how we should perform our hypothesis test. 

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
A quick inspection reveals a similar mean in quality score between the two wines. However, the distrbutions for the white wine appears to be normal while the red wine quality score is close to normal but shows slightly different peak. We do not perform any data transformations but we do check for outliers below in the form of a box-and-whisker plot.

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

There are outliers in both data sets. If we were confident that both data sets were normally distributed we might opt to remove samples past two standard deviations in both directions from the mean. However, we take a more pragmatic approach and remove ouliers via the _IQR Method_. The IQR method removes data _1.5_ times the interquartile range in both directions from the data set. 

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
We quickly check the descriptive statitics of the quality column of the red and white wine data sets after removing the outliers. We also redraw the box-and-whisker plot. 

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
We want to test if there exist a difference in the mean quality score between red and white wine. Even though white wine quality score has a slightly larger average quality score we cannot say for certain that the difference is statistically significant. Therefore, without making any assumptions, we will perform a **two-tailed test**. The null and alternative hypothesis are shown below:  
<p align ="center">
  $$H_0: \mu_{\text{red}} = \mu_{\text{white}} $$
</p>
<p align ="center">
  $$H_A: \mu_{\text{red}} \ne \mu_{\text{white}} $$
</p>

### Variable Selection and Modification 
Traditionally, in A/B testing there is a _control_ and a _treatment_, however, in this case study assigning red and white wine to either group is arbitrary. For our purposes we will treat the red wine as the control and the white wine as the treatment. We will codify red and white wine as binary variables later on. 
  > - 0: Red Wine
  > - 1: White Wine
### Verifying Sample Size 
We are are only testing the wine samples from the north of Portugal. However, for verification purposes we complete a _power analysis_ to verify that our sample size is sufficient. Before we do that we have to make certain assumptions before calculating the _power analysis_. Viewing the current distribution of score, and using some professional judgement, we claim that an average score for wine quality is 5 and anything above 5 is considered _above average_. Lets assume that currently 50% of the red wine is above average and intervention takes place if 55% of white wine is above average. 

  -**Power of the test** $$(1-\beta)$$ We use standard convention and set beta to .2.  
  -**Alpha** $$(\alpha)$$ We use standard convention and set alpha to 0.05  
  -**Effect Size** Discussed above, intervention occurs is there is a 5% difference in quality between the two wines  
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
Before selecting the appropriate method of testing our hypothesis, we must check if certain assumptions are met.
### Normality Assumption (Shapiro-Wilk Test)
The Shapiro-Wilk Test is a commonly used statistical tool to conduct a normality check, in particular, we are testing if the data in our sample is normally distributed. 
<p align="center"> 
The Shapiro-Wilk test statistic $W$ is defined as:

$$\begin{equation}
W = \frac{\left( \sum_{i=1}^{n} a_i x_{(i)} \right)^2}{\sum_{i=1}^{n} (x_i - \bar{x})^2}
\end{equation}$$

where:
 $$x_{(i)}$$ is the $i$-th order statistic (i.e., the $i$-th smallest value in the sample)  
   $\bar{x}$ is the sample mean: $\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$,  
   $a_i$ are constants computed from the means, variances, and covariances of the order statistics of a sample of size $n$ from a standard normal distribution, $n$ is the sample size.  

The numerator can also be written as:

$$\begin{equation}
\left( \sum_{i=1}^{n} a_i x_{(i)} \right)^2 = (\mathbf{a}^T \mathbf{x}_{(sorted)})^2
\end{equation}$$  

The denominator is the sample variance (times $n$), ensuring that $W$ lies in the interval $(0, 1]$.

 A $W$ value close to 1 indicates normality, while smaller values suggest departure from normality. The test's significance is assessed using precomputed critical values or p-values based on $W$ and $n$.
</p>

>Author's note

We are essentially comparing an expected variance by the true variance of our sample, the closer to 1 the more normally distrubuted our data is. However, even if the Shapiro-Wilk Test rejects the null hypothesis that our sample distribution is equal to a normal distribution does not mean we cannot use a T-test, there are limitations to the Shapiro-Wilk test that arise when many values in our sample are equal or if our sample size is too small or too large. We interpret the results with caution as our quality score are interger values from 0 to 10.

#### Shapiro-Wilk Test in Python
The scipy python library we conduct our Shapiro-Wilk Test for us. We conduct a Shapiro-Wilk test for both red and white wine groups from our sample. 
```python
from scipy.stats import shapiro

red_shapiro = shapiro(sample_red['quality'])
white_shapiro = shapiro(sample_white['quality'])
print("Result of Red Wine Shaprio-Wilk Test:",red_shapiro)
print("Results of White Wine Shapiro-Wilk Test:",white_shapiro)
```
  >output is shown below
```
Result of Red Wine Shaprio-Wilk Test: ShapiroResult(statistic=np.float64(0.8339943241222685), pvalue=np.float64(1.5134864505375702e-37))
Results of White Wine Shapiro-Wilk Test: ShapiroResult(statistic=np.float64(0.8511814672686954), pvalue=np.float64(4.9953622517226484e-36))
```
Based on the p-values, we reject the null hypotheis underlying the Shapiro-Wilk Test. There is sufficient evidence to reject the statement that the quality scores from our sample distribution are equal to a normal distribution. 

### Homegeneity of Variances(Levene's Test)
Before proceeding to a T-test we must check another assumption--the variance of the quality scores between our white and red wine sample. We conduct Levene's Test to check that the variance of our two samples are equal.  The formal definition is provided below.
The test statistic $W$ is given by:   
<p align="center"> 
$$W = \frac{(N - k)}{(k - 1)} \cdot \frac{\sum_{i=1}^{k} n_i (\bar{Z}_{i\cdot} - \bar{Z}_{\cdot\cdot})^2}{\sum_{i=1}^{k} \sum_{j=1}^{n_i} (Z_{ij} - \bar{Z}_{i\cdot})^2}$$
</p>
Where:
<p align="center">
$k$ is the number of groups
</p>
<p align="center">
$N$ is the total number of observations across all groups: $N$ = $$\sum_{i=1}^{k} n_i$$
</p>
<p align="center">
  $n_i$ is the number of observations in group $i$
</p>
<p align="center">
$Z_{ij} = |Y_{ij} - \tilde{Y}_i|$, where $Y_{ij}$ is the $j$-th observation in group $i$
</p>
<p align="center">
$\tilde{Y}_i$ is the median (or mean) of group $i$ 
</p>
<p align="center">
$\bar{Z}_{i\cdot}$ is the mean of the $Z_{ij}$ values in group $i$
</p>
<p align="center">
$$\bar{Z}_{\cdot\cdot}$$ is the overall mean of all $Z_{ij}$ values.
</p>
>Author's note
The formulation essentially calculates a variability score to give us an F-Statistics that results in a p-value. The scipy library has a function to calculate the score and corresponding p-value for us.

```python
 #Levene Test
from scipy.stats import levene
#Replace the red and white in the "Color" column with 0 and 1, respectively
combined_sample['Color'] = combined_sample['Color'].replace({'red':0,'white':1} )
test_stat, pvalue = levene(combined_sample[combined_sample["Color"] == 0]["quality"],
                          combined_sample[combined_sample["Color"]==1]["quality"])

print(f'\n     * Levene Stat = {test_stat:.4f},p-value = {pvalue:.4f}')
```
>output shown below
```
     * Levene Stat = 7.7964,p-value = 0.0053
```
Our sample distrbution has failed both the normality test for their distribution and the homegenous variance test. At this point it is not advised to conduct a T-test for the difference of the sample means. We will proceed to the Non-Parametric Testing by performing the Mann-Whitney U-test. 

#### Concluding remarks regarding assumption verification
If our normality and homegeneity assumptions had been met, then we could have proceeded with a T-test for difference in means. If however, normality had been met but not homogeneity, a _Welch's Test_ could have been conducted. However, in our case both assumptions were not met. Our hypothesis testing will be done using non-parametric testing, namely, Mann-Whitney U-Test. 
   >Author's Note (Parametric vs Non-Parametric Testing)
If we made an underlying assumption about the underlying distribution of our sample (Gaussian), we could proceed with parametric testing. However, since we are unable to safely assume any underlying distribution of our sample, we proceed with non-parametric testing.  

For the sake of completeness, we will define and perform a T-test, a Welch's Test, and the Mann-Whitney Test. The testing of our hypothesis testing will come from the Mann-Whitney Test. 

### T-Test (Independent Two Sample T-test)
A Student's T-Test is often used to test is their is a statistically significant difference in response between two groups. In this case study, we could conduct a two-sample T-Test for difference in means. The assumptions have to be met:  
 - The two samples must be equal in size
 - The samples closelt follow a normal distribution
 - The two distribution have the same variance
  
The formal definition is provided below. 
<p align ="center">
$$\begin{equation}
t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
\end{equation}$$
</p>
<p align="center">
  $\bar{X}_1$, $\bar{X}_2$ are the sample means
</p>
<p align="center">
   $s_1^2$, $s_2^2$ are the sample variances  
</p>
<p align="center">
    $n_1$, $n_2$ are the sample sizes  
</p>
We conduct out T-Test in python and import the necessary libraries.

```python
#Conduct t-test
from scipy.stats import ttest_ind
test_stat, pvalue = ttest_ind(combined_sample[combined_sample["Color"] == 0]["quality"],
                          combined_sample[combined_sample["Color"]==1]["quality"],equal_var=True)
print(f'T-Test Stat = {test_stat:.4f}, p-value = {pvalue:.4f}')
````
>output shown below 

```
T-Test Stat = -6.9213, p-value = 0.0000
```
In this hypothetical we reject out null hypothesis and conclude that there is a difference in means between the quality score in red and white wine. However, we will proceed to Welch's Test. 

### Unequal variances T-Test (Welch's t-test)

Welch's T-Test is an augmentation of standard T-Test where the assumption of normality in the distrbution between the sample remains but have unequal variances. The the formulation is almost identical to the Student's T-test where sample variances are **not pooled**. We proceed to conduct the test. 

```python
#Conduct Welch's test
#Same as above, just set equal variance to false 
#Set Variances to false
from scipy.stats import ttest_ind
test_stat, pvalue = ttest_ind(combined_sample[combined_sample["Color"] == 0]["quality"],
                          combined_sample[combined_sample["Color"]==1]["quality"],equal_var=False)
print(f'Welch Test Stat = {test_stat:.4f}, p-value = {pvalue:.4f}')
```
>output shown below
```
Welch Test Stat = -6.3897, p-value = 0.0000
```
In this scenario, we would still reject the null hypothesis if our sample distributions were normal but had unequal variances. 

### Non-Parametric Testing (Mann-Whitney U Test)
Finally, we conduct our Mann-Whitney Test. It is important to note that the Mann-Whitney U Test does **not** directly test for difference in means. Rather, it test if two independent sample distributions are equal--if the central tendency's of two samples are equal or not, based on ordinal data. Recall the intitial observations made in the histograms above, they have a similiar distribution but are slightly different--the red wine has a slight hortizontal shift compared to the white wine. The Mann-Whitney U Test is sensitive to differences in central tendencies only when the distributions between two samples are similarly shaped and spread. The results from the Mann-Whitney U Test can be interpretted as evidence in a shift in _medians_, which for practical purposes suggest a difference in means--a direct result in the shift in distributions.  
The formal definition of the Mann-Whitney U Test is shown below. 
<p align="center">
Let $n_1$ and $n_2$ be the sample sizes of group 1 and group 2, respectively.
</p>
The U statistic for the two groups is defined as:
<p align="center">
$$U_1 = n_1 n_2 + \frac{n_1(n_1 + 1)}{2} - R_1$$
</p>
<p align="center">
$$U_2 = n_1 n_2 + \frac{n_2(n_2 + 1)}{2} - R_2$$
</p>
<p align="center">
Where $R_1$ and $R_2$ are the sums of the ranks for sample 1 and sample 2, respectively.
</p>
The test statistic is:
<p align="center">
$$U = \min(U_1, U_2)$$
</p>
The U statistic is approximated by a normal distribution where:
<p align="center">
$$Z = \frac{U - \mu_U}{\sigma_U}$$
</p>
<p align="center">
$$\mu_U = \frac{n_1 n_2}{2}, \quad \sigma_U = \sqrt{\frac{n_1 n_2 (n_1 + n_2 + 1)}{12}}$$
</p>

  >Author's note  
  
The Mann-Whitney U Test provides an alternative to the Student's T test. An important feature of the test is that the data must be ordinal. The quality score are hierarchical from a scale of 1 to 10, however, many wines share the same quality score since the scores are strictly interger values in the specified range. In order to combat this, a _connected_ rank is calculated. This is done by averaging the rank position for all tied values and using that value in the sum of ranks. An adjusted standard error of U is also calulcated which accounts for rank ties, is it  shown below where $t_i$ is the number of positions sharing the rank $i$ and $k$ is the number of distinct tie groups. 
<p align = "center">
$$\sigma_U = \sqrt{ \frac{n_1 n_2}{12} \left[ (n_1 + n_2 + 1) - \frac{\sum_{i=1}^{k} (t_i^3 - t_i)}{(n_1 + n_2)(n_1 + n_2 - 1)} \right] }$$
</p>

We are now ready to test our null hypothesis. 
```python
#Conduct Mann-Whitney U-Test
from scipy.stats import mannwhitneyu
#set method to asymptotic to account for ties
test_stat, pvalue = mannwhitneyu(combined_sample[combined_sample["Color"] == 0]["quality"],
                          combined_sample[combined_sample["Color"]==1]["quality"],method='asymptotic')
print(f'\n  * Mann-Whitney U Stat = {test_stat:.4f}, p-value = {pvalue:.4f}')
```
>output is shown below
```
  * Mann-Whitney U Stat = 1072000.0000, p-value = 0.0000
```
## Hypothesis Testing Conclusion
The result from Mann-Whitney U Stat is used to test our intial hypothesis. We reject our null hypothesis that mean quality score between red and white wine are equal. There is statistically significant evidence to suggest that they are not equal. 

Our client set out to determine if there was a significant difference in quality between red and white wine. We can safely conclude that there is a difference. Additionally a one sided Mann-Whitney U Test reveals that white wine does  observe a statistically significant higher average quality score compared to red wine.  

The client may choose to supply less red wine options to customers. Futher analysis is required to see which of the physiochemical properties of the wine are the leading factors for determining the quality score. 

## Business Recommedations, SQL, and Visualization (in progress)
While we were able to determine that there is a statistically signficant difference between the quality in red and white wine, we must also provide our client with actionable recommedations based on these findings. 

The next step in our case study is to upload our data in a database and some simple querying in order to make some real word recommendations. 

### Creating A Database in PostgreSQL
The next step is to upload our data into a PostgreSQL database. This will allow us to query the entirety of the database in order to find relationships between quality score and some of the physiochemical attributes of the wine. We will use python and some associated libraries to create and upload the database into PostgreSQL. 

```python
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import sqlalchemy

#Preliminary Confirgurations
postgres_user = "postgres"
#password hidden from demonstration
postgres_password = "*********"
postgres_host = "localhost"
postgres_port = 5432  # Default
new_db_name = "red_white_wine_new"
csv_file_path = "your_data.csv"
table_name = "my_table"
# --------------------------------------

#Connect to the default 'postgres' database to create a new one
conn = psycopg2.connect(
    dbname="postgres",
    user=postgres_user,
    password=postgres_password,
    host=postgres_host,
    port=postgres_port
)
conn.autocommit = True
cur = conn.cursor()

#Create a new database
try:
    cur.execute(f"CREATE DATABASE {new_db_name}")
    print(f"Database '{new_db_name}' created successfully.")
except psycopg2.errors.DuplicateDatabase:
    print(f"Database '{new_db_name}' already exists.")
finally:
    cur.close()
    conn.close()

#Load the CSV into a DataFrame
df = combined_wine

#Connect to the new database with SQLAlchemy
engine = create_engine(
    f"postgresql+psycopg2://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{new_db_name}"
)

#Upload the DataFrame to PostgreSQL
df.to_sql(table_name, engine, index=False, if_exists='replace')  # 'append' if you want to keep existing data

print(f"Data uploaded to table '{table_name}' in database '{new_db_name}'.")
```
>output shown below
```
Data uploaded to table 'my_table' in database 'red_white_wine_new'.
```

