import torch
import numpy as np
import matplotlib.pyplot as plt



def plot (i,y_i,color):
    # plot real values
    plt.plot(np.arange(999),y_i[:999],color=color,linewidth=2.0) 
    # plot expected values
    plt.plot(np.arange(999,1999),y_i[999:],color=color,linewidth=1.0,ls='--')

