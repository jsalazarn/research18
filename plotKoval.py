import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()

def plotK(set1, set2):
    
    fig, ax = plt.subplots(2, 1)
    rows , columns = set1.shape
    
    for i in np.arange(int(columns / 2)):
        ax[0].plot(set1.iloc[:, 2 * i], set1.iloc[:, 2 * i + 1], 'ro',
                 markersize=4, alpha=0.8)
        ax[0].plot(set2.iloc[:, 2 * i], set2.iloc[:, 2 * i + 1], 'bs',
                 markersize=4, alpha=0.8)
        
        ax[1].plot(set1.iloc[:, 2 * i], set1.iloc[:, 2 * i + 1], 'ro',
                 markersize=4, alpha=0.8)
        ax[1].plot(set2.iloc[:, 2 * i], set2.iloc[:, 2 * i + 1], 'bs',
                 markersize=4, alpha=0.8)
        
    ax[0].set(xscale='log', yscale='log', xlabel='(1-C)/C', ylabel='(1-F)/F',)
    ax[0].set_title('Koval Plots', weight='bold', size=22)
    ax[0].xaxis.grid(True, which='minor', linestyle='-', linewidth=0.5,)
    ax[0].yaxis.grid(True, which='minor', linestyle='-', linewidth=0.5,);
    
    ax[1].set(xlabel='(1-C)/C', ylabel='(1-F)/F',)
    ax[1].set_title('Koval Plots', weight='bold', size=22)
    ax[1].xaxis.grid(True, which='minor', linestyle='-', linewidth=0.5,)
    ax[1].yaxis.grid(True, which='minor', linestyle='-', linewidth=0.5,);