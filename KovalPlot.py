import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

def plotkv(plt_reb1, plt_reb2):
    """
    This function plots the Koval data.
    
    Parameters
    ----------
    plt_reb1 : First matrix data that contains all (1-C)/C and (1-F)/F
    plt_reb2 : Second matrix data that contains all (1-C)/C and (1-F)/F
    """
    

    fig, ax = plt.subplots()
    rows , columns = plt_reb1.shape
    
    for i in np.arange(int(columns / 2)):
        plt.plot(plt_reb1.iloc[:, 2 * i], plt_reb1.iloc[:, 2 * i + 1], 's',
                 markersize=4, alpha=0.6)
        plt.plot(plt_reb2.iloc[:, 2 * i], plt_reb2.iloc[:, 2 * i + 1], 'o',
                 markersize=4, alpha=0.6)
        
    ax.set(xscale='log', yscale='log', xlabel='(1-C)/C', ylabel='(1-F)/F',
           xlim=(1e-3, 1e3), ylim=(1e-6, 1e2))
    
    ax.set_title('Koval Plots', weight='bold', size=22)
    ax.legend()
    
    ax.xaxis.grid(True, which='minor', linestyle='-', linewidth=0.5,)
    ax.yaxis.grid(True, which='minor', linestyle='-', linewidth=0.5,);