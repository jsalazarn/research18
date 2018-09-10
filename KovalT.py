import pandas as pd
import numpy as np
from scipy import stats

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
    
    plot_Koval = pd.DataFrame(np.zeros((sizen, iteras * 2)))
    
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
        x.where(x > 1e-8, 0, inplace=True) # replace values < 1e-8 with 0
        x.drop(['interst vel'], axis=1, inplace=True)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
        x['(1-C)/C'], x['(1-F)/F'])
        
        # Store results of linear regression
        KLR.iloc[itera, 0:2] = (x['porosity'].mean(), x['permeability'].mean())
        KLR.iloc[itera, 2:] = slope, intercept, p_value, r_value ** 2, std_err
        
        # Store Koval data sets for plotting
        plot_Koval.iloc[:, 2 * itera] = x['(1-C)/C'].values
        plot_Koval.iloc[:, 2 * itera + 1] = x['(1-F)/F'].values
        
    return KLR, plot_Koval