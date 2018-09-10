import numpy as np
import pandas as pd
from scipy import stats

def KovalF(mean_log, sd_log, mean_nor, sd_nor, sizen, iteras, k_fr):
    """
    This function computes and stores the linear regression results of
    the Koval plot. Parameters:
    Permeability: lognormal; mean and standard deviation [md]
    Porosity: normal; mean and standard deviation
    sizen: number of points of the data sets
    iteras: the number of linear regression analyses to be performed
    k_fr: the permeability value for a fracture (in order of darcies)
    """
    
    import random
    
    KLR = pd.DataFrame(np.zeros((iteras, 7)),
                      columns=['Porosity Mean', 'Permeability Mean', 'Slope', 'Intercept',
                               'P value', 'R^2', 'Standard Error']) # initializing DataFrame of results
    
    plot_Koval = pd.DataFrame(np.zeros((sizen, iteras * 2)))
    
    # kphi = pd.DataFrame(np.zeros((sizen, iteras * 2)))
    
    for itera in np.arange(iteras):
        # Compute the mean and standard deviation of lognormal pdf in terms of
        # the normal distribution
        normal_std = np.sqrt(np.log(1 + (sd_log/mean_log)**2))
        normal_mean = np.log(mean_log) - normal_std **2 / 2
        
        # Create DataFrame and permeability and porosity columns
        perm1 = np.random.lognormal(normal_mean, normal_std, size=sizen) # first data set of permeability
        
        perm2 = np.random.random(*perm1.shape) * (k_fr - sd_log) + (k_fr + sd_log)
        # This creates a data set of fractured permeability values: k_fracture - std dev k < k < k_fracture + std dev k
        
        ten_per = int(np.floor(sizen * 0.1)) # ten percent = 0.1
        perm1[np.random.randint(0, sizen, size=ten_per)] = random.sample(list(perm2), ten_per)
        # The data set perm1 is updated by randomly modifying 10% of its values to high permeability values
        
        x = pd.DataFrame(perm1, columns=['permeability'])
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
        #x.where(x > 0, 0, inplace=True) # if there are negative values in the DataFrame, replace them with 0
        x.where(x > 1e-8, 0, inplace=True) # replace values < 1e-8 with 0
        #x.drop(['interst vel'], axis=1, inplace=True)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
        x['(1-C)/C'], x['(1-F)/F'])
        
        # Store linear regression results
        KLR.iloc[itera, 0:2] = (x['porosity'].mean(), x['permeability'].mean())
        # Store linear regression results
        KLR.iloc[itera, 2:] = slope, intercept, p_value, r_value ** 2, std_err
        
        # Store Koval data sets for plotting.
        # Even columns: (1-C)/C; Odd columns: (1-F)/F
        plot_Koval.iloc[:, 2 * itera] = x['(1-C)/C'].values
        plot_Koval.iloc[:, 2 * itera + 1] = x['(1-F)/F'].values
        
        # Store permeability values of porosity and permeability
        # kphi.iloc[:, 2 * itera] = x['porosity']
        # kphi.iloc[:, 2 * itera + 1] = x['permeability']
        
    return KLR, plot_Koval