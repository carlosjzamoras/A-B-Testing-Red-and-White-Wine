#Red and White Wine Data Set
#Goal: Is there a difference in the quality of wine based on color?
#Data Processing and create SQL Database
# Visualize interesting factors of red and white wine
#Display on project header
#Perform A/B testing 
#Concluding Remarks 

#Load Data set and clean 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
red_wine_df = pd.read_csv('/Users/carloszamora/Downloads/wine+quality/winequality-red.csv',sep=";")
white_wine_df = pd.read_csv('/Users/carloszamora/Downloads/wine+quality/winequality-white.csv',sep=";")
# Create additional column in both dataframes specifying the color 
red_wine_df['Color'] = 'red'
white_wine_df['Color'] = 'white'
#Combine the data frames 
frames = [red_wine_df,white_wine_df]
combined_wine = pd.concat(frames)

#Simple EDA 
#Look at summary statistics of for quaility of for both groups
combined_wine['quality'].hist()
plt.show()
#Looks narrow- clustered around the mean with low variance
#Print Summary Statistics for each group 
print(red_wine_df['quality'].describe())
print(white_wine_df['quality'].describe())
#Exam the distribution of quality based on color

fig,axes = plt.subplots(nrows=1,ncols=2,figsize = (12,4))
axes[0].hist(red_wine_df['quality'],color='Red',edgecolor='black',bins=6)
axes[0].set_title('Red Wine')

axes[1].hist(white_wine_df['quality'],color='White',edgecolor='black',bins=6)
axes[1].set_title('White Wine')

# #Labels and title 
for ax in axes:
    ax.set_xlabel('Quality Score')
    ax.set_ylabel('Frequency')
plt.tight_layout()
plt.suptitle("Distribution of Wine Quality")
plt.show()

# # Check for outliers
# # Box and Whisker plot 
fig, axes = plt.subplots(1,2)
axes[0].boxplot(red_wine_df['quality'])
axes[0].set_xlabel('Red Wine')
axes[0].set_xticks([])

axes[1].boxplot(white_wine_df['quality'])
axes[1].set_xlabel('White Wine')
axes[1].set_xticks([])

# #Labels and title 
for ax in axes:
    ax.set_ylabel("Quality Score")
plt.suptitle("Box-and-Whisker Plot of Wine Quality")
plt.show()

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

# #Quickly Check the descriptive statistics of red and white wine
print(red_wine_df['quality'].describe())
print(white_wine_df['quality'].describe())
# #Print box and whisker plot without outliers 

fig, axes = plt.subplots(1,2)
axes[0].boxplot(red_wine_df['quality'])
axes[0].set_xlabel('Red Wine')
axes[0].set_xticks([])

axes[1].boxplot(white_wine_df['quality'])
axes[1].set_xlabel('White Wine')
axes[1].set_xticks([])

# #Labels and title 
for ax in axes:
    ax.set_ylabel("Quality Score")
plt.suptitle("Box-and-Whisker Plot of Wine Quality")
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
                    hspace=0.4, wspace=0.3)
plt.show()


# #Hypothesis Testing
# from scipy.stats import (
#     ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, 
#     pearsonr, spearmanr, kendalltau, f_oneway, kruskal
# )
# from statsmodels.stats.proportion import proportions_ztest

# def hypothesis_testing(df, group_col, test_value, control_value,target_col):
#     #H0: There is no difference in quality between white and red wine
#     #H1: Alternative hypothesis 
#     #Reject the H0 if p-value <0.05

#     #Normality Assumption 
#     print("-"*100)
#     print("Assumption Check Step 1")
#     print("Shapiro-Wilk Test")
#     test_stat_test , pvalue_test = shapiro(df[df[group_col] == test_value][target_col])
#     test_control_test, pvalue_control = shapiro(df[df[group_col] == control_value][target_col])
    
#     if pvalue_test >0.05 and pvalue_control> 0.05:
#         print("\n      * Normality assumption is met.")
#         print(f'\n    * Test Group State = {test_stat_test:.4f}, p-value = {pvalue_test:.4f}' )
#         print(f'\n    * Control Group State = {test_control_test:.4f}, p-value = {pvalue_control:.4f}' )
#         normality_assumption = True
#     else:
#         print("\n      * Normality assumption is not met.")
#         print(f'\n    * Test Group State = {test_stat_test:.4f}, p-value = {pvalue_test:.4f}' )
#         print(f'\n    * Control Group State = {test_control_test:.4f}, p-value = {pvalue_control:.4f}' )
#         normality_assumption = False
#     #Checking the homogeneity of Variances
#     print("-"*100)
#     print('Assumpttion Check Step  2')
#     print("Levene's Test")
    
#     test_stat, pvalue = levene(df[df[group_col] == test_value][target_col],
#                                df[df[group_col]==control_value][target_col])
#     if pvalue > 0.05:
#         variance_assumption = True
#         print("\n      * Variances are homogeneous.")
#         print(f'\n     * Levene Stat = {test_stat:.4f},p-value = {pvalue:.4f}')
#     else:
#         variance_assumption = False
#         print("\n      * Variances are not homogeneous.")
#         print(f'\n     * Levene Stat = {test_stat:.4f},p-value = {pvalue:.4f}')
#     #Parametric Test: T-test 
#     if normality_assumption and variance_assumption:
#         print('-' * 100)
#         print('Assumptions met, proceed with independent T-test')
#         test_stat, pvalue = ttest_ind(df[df[group_col] == test_value][target_col], 
#                                       df[df[group_col] == control_value][target_col],
#                                       equal_var=True)
#         if pvalue > 0.05:
#             print('Fail to Reject H0')
#             print(f'T-test Stat = {test_stat:.4f}, p-value = {pvalue:.4f}')
#         else:
#             print("Reject H0")
#             print(f'T-test Stat = {test_stat:.4f}, p-value = {pvalue:.4f}')
#     elif normality_assumption and not variance_assumption:
#         print("-"*100)
#         print('Normality met but variances are not homogenenous, proceed to Welch Test')
#         test_stat, pvalue = ttest_ind(df[df[group_col] == test_value][target_col], 
#                                       df[df[group_col] == control_value][target_col],
#                                       equal_var=False)
#         if pvalue > 0.05:
#             print("Fail to Reject H0")
#             print(f'T-Test Stat = {test_stat:.4f}, p-value = {pvalue:.4f}')
#         else:
#             print("Reject H0")
#             print(f'T-Test Stat = {test_stat:.4f}, p-value = {pvalue:.4f}')
    
#     # Non-Parametric Test: Mann-Whitney U Test
#     else:
#         print("-" * 100)
#         print("Assumptions not met, performing Mann-Whitney U Test (Non-Parametric)")
        
#         test_stat, pvalue = mannwhitneyu(df[df[group_col] == test_value][target_col], 
#                                          df[df[group_col] == control_value][target_col])
    
#         if pvalue > 0.05:
#             print("\n  * Fail to Reject H0")
#             print(f'\n  * Mann-Whitney U Stat = {test_stat:.4f}, p-value = {pvalue:.4f}')
#         else:
#             print("\n  * Reject H0")
#             print(f'\n  * Mann-Whitney U Stat = {test_stat:.4f}, p-value = {pvalue:.4f}')

        
#Replace red and white wine with dummy variables 
# combined_wine['Color'] = combined_wine['Color'].replace({'white':1,'red':0} )
# hypothesis_testing(combined_wine,'Color',1,0,'quality')

#Create power tesst to get effective size 
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

#Take a random sample of 1565 from both groups 
sample_red = red_wine_df.sample(n=1565)
sample_white=white_wine_df.sample(n=1565)
frames=[sample_red,sample_white]
#Combined sample will be used later 
combined_sample = pd.concat(frames)

# ##Shapiro Wilk Test
from scipy.stats import shapiro

red_shapiro = shapiro(sample_red['quality'])
white_shapiro = shapiro(sample_white['quality'])
print("Result of Red Wine Shaprio-Wilk Test:",red_shapiro)
print("Results of White Wine Shapiro-Wilk Test:",white_shapiro)

#Levene Test
from scipy.stats import levene

#Replace the red and white in the "Color" column with 0 and 1, respectively

combined_sample['Color'] = combined_sample['Color'].replace({'red':0,'white':1} )
test_stat, pvalue = levene(combined_sample[combined_sample["Color"] == 0]["quality"],
                          combined_sample[combined_sample["Color"]==1]["quality"])

print(f'\n     * Levene Stat = {test_stat:.4f},p-value = {pvalue:.4f}')

#Conduct t-test
from scipy.stats import ttest_ind
test_stat, pvalue = ttest_ind(combined_sample[combined_sample["Color"] == 0]["quality"],
                          combined_sample[combined_sample["Color"]==1]["quality"],equal_var=True)
print(f'T-Test Stat = {test_stat:.4f}, p-value = {pvalue:.4f}')

#Conduct Welch's test
#Same as above, just set equal variance to false 
#Set Variances to false
from scipy.stats import ttest_ind
test_stat, pvalue = ttest_ind(combined_sample[combined_sample["Color"] == 0]["quality"],
                          combined_sample[combined_sample["Color"]==1]["quality"],equal_var=False)
print(f'Welch Test Stat = {test_stat:.4f}, p-value = {pvalue:.4f}')

#Conduct Mann-Whitney U-Test
from scipy.stats import mannwhitneyu
from statsmodels.stats.proportion import proportions_ztest
#set method to asymptotic to account for ties
test_stat, pvalue = mannwhitneyu(combined_sample[combined_sample["Color"] == 0]["quality"],
                          combined_sample[combined_sample["Color"]==1]["quality"],method='asymptotic',alternative='less')
print(f'\n  * Mann-Whitney U Stat = {test_stat:.4f}, p-value = {pvalue:.4f}')
