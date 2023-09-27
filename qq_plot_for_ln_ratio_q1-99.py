import pandas as pd
import os
import re
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import numbers

import numpy as np
import matplotlib.pyplot as plt

def fig_distribution(data,N1,CV,MoM,item):
    # Create a histogram
    fig, ax = plt.subplots()
    plt.hist(data, bins=20, color='blue', alpha=0.7)
    plt.title(f'Distribution_{N1}_{CV}_{MoM}_{item}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    fig.savefig(f"Distribution_{N1}_{CV}_{MoM}_{item}_20230927.png")
    # Display the histogram
    # plt.show()

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
    X = rng.normal(size=nMonte)  # Sample 2: Y ~ t(5)
    X = np.percentile(X, percentiles, interpolation='midpoint')
    # Create a normal quantile plot
    fig, ax = plt.subplots()

    # Plot CV1
    Y1 = np.percentile(CV1, percentiles, interpolation='midpoint')

    ax.scatter(X, Y1, color='red', label=CV1_label)

    # Plot regression line for CV1
    slope, intercept, r_value, _, _ = stats.linregress(X, Y1)
    ax.plot(X, slope * X + intercept, color='red', linestyle='--')
    ax.text(0.65, 0.4, f'Y = {slope:.2f}X + {intercept:.2f}', transform=ax.transAxes, color='red')
    ax.text(0.65, 0.35, f'R^2 = {r_value**2:.2f}', transform=ax.transAxes, color='red')

    # Plot CV2
    Y2 = np.percentile(CV2, percentiles, interpolation='midpoint')
    ax.scatter(X, Y2, color='blue', label=CV2_label)

    # Plot data (optional)
    # ax.scatter(q_theoretical, q_data, color='green', label='Data')
    
    # Plot regression line for CV2
    slope, intercept, r_value, _, _ = stats.linregress(X, Y2)
    ax.plot(X, slope * X + intercept, color='blue', linestyle='--')
    ax.text(.65, 0.25, f'Y = {slope:.2f}X + {intercept:.2f}', transform=ax.transAxes, color='blue')
    ax.text(.65, 0.2, f'R^2 = {r_value**2:.2f}', transform=ax.transAxes, color='blue')

    # Customize the plot
    ax.set_title(f"Quantile-Quantile Plot_N{N}_{MoM}_{item}")
    ax.set_xlabel("Theoretical Percentiles")
    ax.set_ylabel("Observed Percentiles")
    ax.legend()

    # Save the plot as an image file
    fig.savefig(f"quantile-quantile_plot_N{N}_{MoM}_{item}_20230927.png")

    # Show the plot
    # plt.show()

# Define the directory where your text files are located
directory = "C:/Users/User/MC_sim_2SD/Weibull_no_moments_20230927_MeanTimeScale_1"

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

# normal_quantile_plot(CV1, CV2, 'CV=0.15', 'CV=0.5', N, MoM, 'ln_ratio')

fig_distribution(CV1,N,'CV=0.15',MoM,'ln_ratio')
fig_distribution(CV2,N,'CV=0.5',MoM,'ln_ratio')

quit()