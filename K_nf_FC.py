import pandas as pd
import numpy as np

def FCK_nf(mean_perm, sd_perm, mean_phi, sd_phi, sizen, iteras):
    """
    This function computes lognormal permeability. It computes and displays the
    F and C capacities values and the (1-F)/F and (1-C)/C terms.
    
    It helps to determine the deviation from a straight line of the Koval plot
    and if the theory could identify naturally fractured reservoirs.
    
    References
    ----------------------------------------------
    mean_perm = mean of permeability
    sd_perm = sd of permeability
    """
    
    KLR = pd.DataFrame(np.zeros((sizen, 7)),
                      columns=['Porosity', 'Permeability', 'k/phi', 'C', 'F', '(1-C)/C',
                               '(1-F)/F']) # initializing DataFrame of results
    
    plot_Koval = pd.DataFrame(np.zeros((sizen, iteras * 2)))
    
    # kphi = pd.DataFrame(np.zeros((sizen, iteras * 2)))
    
    for itera in np.arange(iteras):
        # Compute the mean and standard deviation of lognormal pdf in terms of
        # the normal distribution
        normal_std = np.sqrt(np.log(1 + (sd_perm/mean_perm)**2))
        normal_mean = np.log(mean_perm) - normal_std **2 / 2
        
        # Create DataFrame and permeability and porosity columns
        perm1 = np.random.lognormal(normal_mean, normal_std, size=sizen) # first data set of permeability
        
        x = pd.DataFrame(perm1, columns=['permeability'])
        np.random.seed(100)
        x['porosity'] = np.random.normal(mean_phi, sd_phi, size=sizen)
        
        # Obtain the total sum of both permeability and porosity
        sumk = x['permeability'].sum()
        sumphi = x['porosity'].sum()
        x['interst vel'] = x['permeability'] / x['porosity'] # Interstitial velocity
        
        # Sort the interstitial velocity in descending order
        x.sort_values(by=['interst vel'], ascending=False, inplace=True)
        
        k1 = np.cumsum(pd.DataFrame(x['permeability']))
        phi1 = np.cumsum(pd.DataFrame(x['porosity']))
        F = k1 / sumk
        C = phi1 / sumphi
        
        
        x['(1-F)/F'] = (1 - F) / F
        x['(1-C)/C'] = (1 - C) / C
        #x.where(x > 0, 0, inplace=True) # if there are negative values in the DataFrame, replace them with 0
        x.where(x > 1e-8, 0, inplace=True) # replace values < 1e-8 with 0
        #x.drop(['interst vel'], axis=1, inplace=True)
        
        KLR.iloc[:, itera * 7] = x['porosity'].values
        KLR.iloc[:, itera * 7 + 1] = x['permeability'].values
        KLR.iloc[:, itera * 7 + 2] = x['interst vel'].values
        KLR.iloc[:, itera * 7 + 3] = C['porosity'].values
        KLR.iloc[:, itera * 7 + 4] = F['permeability'].values
        KLR.iloc[:, itera * 7 + 5] = x['(1-C)/C'].values
        KLR.iloc[:, itera * 7 + 6] = x['(1-F)/F'].values
                
        # Store Koval data sets for plotting.
        # Even columns: (1-C)/C; Odd columns: (1-F)/F
        plot_Koval.iloc[:, 2 * itera] = x['(1-C)/C'].values
        plot_Koval.iloc[:, 2 * itera + 1] = x['(1-F)/F'].values
        
    KLR['Reservoir'] = 'non-fractured'
    return KLR, plot_Koval