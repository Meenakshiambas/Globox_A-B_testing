#!/usr/bin/env python
# coding: utf-8

# # GloBox Data analysis and A/B testing For Control and Treatment Group 

# In[201]:


#importing libraries


# In[202]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.stats import ttest_ind


# In[203]:


# Downloading Globox 


# In[204]:


globox_data = pd.read_csv('Globox_data_for_python.csv')


# In[205]:


globox_data


# # Total Spent Analysis

# In[206]:


Spent_A = globox_data.query("group=='A'")["total_spent"]


# In[207]:


Spent_A 


# In[208]:


Spent_B = globox_data.query("group=='B'")["total_spent"]


# In[209]:


Spent_B


# ### Plotting total spent by group

# In[210]:


plt.figure(figsize=(8,6))
x=globox_data['group']
y=globox_data['total_spent']
sns.barplot(x, y, ci=False)
plt.title('Total Spent Analysis by group', pad=20)
plt.xlabel('Group', labelpad=20)
plt.ylabel('Total_spent', labelpad=20);


# In[211]:


avg_spent_per_user_A_B = globox_data.groupby('group')['total_spent'].mean()


# In[212]:


avg_spent_per_user_A_B


# In[213]:


stdev_spent_per_user_A_B = globox_data.groupby('group')['total_spent'].std()


# In[214]:


stdev_spent_per_user_A_B


# In[215]:


# A/B testing on total_spent for control and treatment group


# ## Hypothesis Test, to test that there is diffrence between mean total spent per user in Control and Treatment groups
# 
# ## Null Hypothesis: H0 : There is no significant diffrence between mean total spent by per user in Control and Treatment groups
# 
# ## Alternate Hypothesis: H1 : There is a significant diffrence between mean total spent by per user in Control and Treatment groups

# In[216]:


#Seperating data into two groups


# In[217]:


Group_A = globox_data[globox_data.group=='A']
Group_B = globox_data[globox_data.group=='B']


# In[218]:


Group_A


# In[219]:


Group_B


# In[220]:


# T-test for t-statistics and determining pvalue


# In[221]:


ttest_ind(Group_A.total_spent,Group_B.total_spent,equal_var=False)


# ### A/B testing Result

# ## pvalue is greater than 0.05 that is 0.944 approximately, We fail to reject the null hypothesis that there is no diffrence in mean amount spent per user between the control and treatment group.

# ### Confidence interval Calculation
# 

# In[222]:


Confidence_interval_A = st.t.interval(alpha=0.95, df=len(Spent_A)-1, loc=np.mean(Spent_A), scale=st.sem(Spent_A)) 


# In[223]:


Confidence_interval_A


# In[224]:


Confidence_interval_B = st.t.interval(alpha=0.95, df=len(Spent_B)-1, loc=np.mean(Spent_B), scale=st.sem(Spent_B)) 


# In[225]:


Confidence_interval_B


# # Conversion Rate Analysis

# In[226]:


count_of_user_A_B = globox_data.groupby('group')['uid'].count()


# In[227]:


count_of_user_A_B


# In[228]:


Converted_A = globox_data.query('group=="A" & total_spent!= 0 ')


# In[229]:


Converted_A


# In[230]:


Converted_B = globox_data.query('group=="B" & total_spent!= 0 ')


# In[231]:


Converted_B


# ## Hypothesis Test to determine that there is diffrence between conversion_rate by in Control and Treatment groups
# 
# ## Null Hypothesis: H0 : There is no significant diffrence between conversion_rate in Control and Treatment groups
# 
# ## Alternate Hypothesis: H1 : There is significant diffrence between conversion_rate in Control and Treatment groups

# In[232]:


Conversion_rate_A = (955/24343)*100
Conversion_rate_A


# In[233]:


Conversion_rate_B = (1139/24600)*100
Conversion_rate_B


# In[234]:


globox_data['converted'] = np.where(globox_data['total_spent'] != 0, 1, 0)


# In[235]:


globox_data


# In[236]:


conversion_rate = globox_data.groupby('group')['converted']

std_p = lambda x: np.std(x, ddof=0)              # Std. deviation of the proportion
se_p = lambda x: st.sem(x, ddof=0)            # Std. error of the proportion (std / sqrt(n))

conversion_rate = conversion_rate.agg([np.mean, std_p, se_p])
conversion_rate.columns = ['conversion_rate', 'std_deviation', 'std_error']


conversion_rate.style.format('{:.3f}')


# ### Plotting conversion rate for group A and B

# In[237]:


plt.figure(figsize=(8,6))
x=globox_data['group']
y=globox_data['converted']
sns.barplot(x, y, ci=False)
plt.title('Conversion rate analysis by group', pad=20)
plt.xlabel('Group', labelpad=20)
plt.ylabel('Converted (proportion)', labelpad=20);


# In[238]:


#Z-test for z-statistics and determining p-value and confidence-interval


# In[239]:


from statsmodels.stats.proportion import proportions_ztest, proportion_confint
A_converted = globox_data[globox_data['group'] == 'A']['converted']
B_converted = globox_data[globox_data['group'] == 'B']['converted']
n_A = A_converted.count()
n_B = B_converted.count()
total = [A_converted.sum(), B_converted.sum()]
n_AB = [n_A, n_B]

z_stat, pval = proportions_ztest(total, nobs = n_AB)
(lower_con, lower_treat), (upper_con, upper_treat) = proportion_confint(total, nobs=n_AB, alpha=0.05)

print(f'z statistic: {z_stat:.3f}')
print(f'p-value: {pval:.4f}')
print(f'ci 95% for control group: [{lower_con:.3f}, {upper_con:.3f}]')
print(f'ci 95% for treatment group: [{lower_treat:.3f}, {upper_treat:.3f}]')


#  
#  ## p = 0.0001, statistically significant. We can reject the null hypothesis that there is no      difference in the user conversion rate between the control and treatment.
# 

# # END

# In[ ]:




