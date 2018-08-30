
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')

def KovalT(mean_log, sd_log, mean_nor, sd_nor, sizen, iteras):
    """
    This function computes and stores the linear regression results of
    the Koval plot. Parameters:
    Permeability: lognormal; mean and standard deviation
    Porosity: normal; mean and standard deviation
    sizen: number of points of the data sets
    iteras: the number of linear regression analysis to be performed.
    """
    
    KLR = pd.DataFrame(np.zeros((iteras, 7)),
                      columns=['Porosity Mean', 'Permeability Mean', 'Slope', 'Intercept',
                               'P value', 'R^2', 'Standard Error']) # initializing DataFrame of results
    
    for itera in np.arange(iteras):
        # Compute the mean and standard deviation of lognormal pdf in terms of
        # the normal distribution
        normal_std = np.sqrt(np.log(1 + (sd_log/mean_log)**2))
        normal_mean = np.log(mean_log) - normal_std**2 / 2
        
        # Create DataFrame and permeability and porosity columns
        x = pd.DataFrame(np.random.lognormal(normal_mean, normal_std, size=sizen),
                         columns=['permeability'])
        x['porosity'] = np.random.normal(mean_nor, sd_nor, size=sizen)
        
        # Obtain the total sum of both permeability and porosity
        sumk = x['permeability'].sum()
        sumphi = x['porosity'].sum()
        x['interst vel'] = x['permeability'] / x['porosity'] # Interstitial velocity
        
        # Sort the interstitial velocity in descending order
        x.sort_values(by=['interst vel'], ascending=False, inplace=True)
        
        k1 = np.cumsum(pd.DataFrame(x['permeability']))
        phi1 = np.cumsum(pd.DataFrame(x['porosity']))
        x['(1-F)/F'] = (sumk / k1) - 1
        x['(1-C)/C'] = (sumphi / phi1) - 1
        x.where(x > 0, 0, inplace=True) # if there are negative values in the DataFrame, replace them with 0
        x.drop(['interst vel'], axis=1, inplace=True)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
        x['(1-C)/C'], x['(1-F)/F'])
        
        KLR.iloc[itera, 0:2] = (x['porosity'].mean(), x['permeability'].mean())
        KLR.iloc[itera, 2:] = slope, intercept, p_value, r_value ** 2, std_err
        
    return KLR


# In[2]:


KLR = KovalT(100, 85, 0.27, 0.04, 2000, 10000)


# In[3]:


KLR.describe()


# In[6]:


fig, ax = plt.subplots(2)
plt.style.use('seaborn-whitegrid')
ax[0].hist(KLR['Slope'], bins=100)
ax[1].boxplot(KLR['Slope'], vert=False);


# In[13]:


fig, ax = plt.subplots(1)
plt.style.use('seaborn-whitegrid')

ax.plot(KLR['P value'], KLR['Standard Error'], 'o')
ax.set(xlabel='P value', ylabel='Standard Error');


# In[101]:


result = stats.anderson(KLR['Slope'], dist='logistic')


# In[102]:


result = stats.anderson(KLR['Slope'], dist='norm')
print('Statistic: %.3f' % result.statistic)
p = 0
for i in range(len(result.critical_values)):
    sl, cv = result.significance_level[i], result.critical_values[i]
    if result.statistic < result.critical_values[i]:
        print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
    else:
        print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))


# In[98]:


result.statistic


# In[100]:


result.critical_values

