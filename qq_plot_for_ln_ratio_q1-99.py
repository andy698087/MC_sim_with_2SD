import pandas as pd
import os
import re
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import numbers

import numpy as np
import matplotlib.pyplot as plt


def normal_quantile_plot(CV1, CV2, CV1_label, CV2_label, nMonte, MoM, item):
    # Desired percentiles
    percentiles = list(range(1, 100))

    # Compute quantiles for data and theoretical distribution
    # q_theoretical = np.percentile(stats.norm.rvs(size=10000), percentiles)  # based on a normal distribution
    # q_data = np.percentile(data, percentiles)
    
    # Setup
    rng = np.random.RandomState(0)  # Seed RNG for replicability

    # Generate data
    # x = rng.normal(size=N)  # Sample 1: X ~ N(0, 1)
    Y = rng.standard_t(df=5, size=nMonte)  # Sample 2: Y ~ t(5)
    Y = np.percentile(Y, percentiles, interpolation='nearest')
    # Create a normal quantile plot
    fig, ax = plt.subplots()

    # Plot CV1
    X1 = np.percentile(CV1, percentiles, interpolation='nearest')

    ax.scatter(X1, Y, color='red', label=CV1_label)

    # Plot CV2
    X2 = np.percentile(CV2, percentiles, interpolation='nearest')
    ax.scatter(X2, Y, color='blue', label=CV2_label)

    # Plot data (optional)
    # ax.scatter(q_theoretical, q_data, color='green', label='Data')

    # Customize the plot
    ax.set_title(f"Normal Quantile Plot_N{N}_{MoM}_{item}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()

    # Save the plot as an image file
    fig.savefig(f"quantile-quantile_plot_N{N}_{MoM}_{item}_20230927.png")

    # Show the plot
    # plt.show()

# Define the directory where your text files are located
directory = "C:/Users/User/MC_sim_2SD/Weibull_no_moments_20230927"

# Get a list of files that match the pattern in the directory
matching_files = [file for file in os.listdir(directory) if file.startswith("Weibull_GPMMC_nMonte_100000_N") and file.endswith("ts.csv")]
# print(matching_files)
# Initialize an empty list to store data from all files

data = {'MethodOfMoments':[], 'nMonte': [], 'N': [], 'CV': [], 
        'ln_ratio': []}

# Loop through the matching files and extract data
for filename in matching_files:
    filename_frag = filename.rstrip('.csv').split('_')
    nMonte = int(filename_frag[3])
    N = int(filename_frag[5])
    CV = float(filename_frag[7])
    MoM = '_'.join(filename_frag[8:])

    if CV in [0.15,0.5] and  N in [25]:
        
            
        data['MethodOfMoments'].append(MoM)
        data['nMonte'].append(nMonte)
        data['N'].append(N)
        data['CV'].append(CV)

        # Extract relevant information from each file
        df = pd.read_csv(os.path.join(directory,filename))
        data['ln_ratio'].append(df['ln_ratio'])

df = pd.DataFrame(data)
CV1 = df[df['CV'] == 0.15]['ln_ratio'].to_list()
CV2 = df[df['CV'] == 0.5]['ln_ratio'].to_list()
N = df[df['CV'] == 0.15]['N'][0]
MoM = df[df['CV'] == 0.15]['MethodOfMoments'][0]
# normal_quantile_plot(data,N1,CV,MethodOfMoments,str(col))
# prob_plot(data,N1,CV,MethodOfMoments,str(col))
# fig_distribution(data,N1,CV,MethodOfMoments,str(col))
normal_quantile_plot(CV1, CV2, 'CV=0.15', 'CV=0.5', N, MoM, 'ln_ratio')


quit()