

"""
rSampleMeanLogScale1
rSampleSDLogScale1
rSampleMeanLogScale2
rSampleSDLogScale2
lower_bound_SE
upper_bound_SE
ln_ratio
se_ln_ratio
percentile_2_5
percentile_97_5
intervals_include_zero
P_value
percentile_2_5 > 0
percentile_97_5 < 0
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
import os
import re
import pandas as pd

def prob_plot(data, N1,CV,MoM,item):
        
    # Create the probability plot
    res = stats.probplot(data, plot=plt)
    plt.title(f'Probability Plot_{N1}_{CV}_{MoM}_{item}')

    # Save the plot as a PNG image
    plt.savefig(f'probability_plot__{N1}_{CV}_{MoM}_{item}.png')


def normal_quantile_plot(data,N1,CV,MoM,item):

    # Create a normal quantile plot
    fig, ax = plt.subplots()
    stats.probplot(data, plot=ax)

    # Customize the plot (optional)
    ax.set_title("Normal Quantile Plot_{N1}_{CV}_{MoM}_{item}")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")

    # Save the plot as an image file (e.g., PNG)
    fig.savefig(f"Random Number Distribution_{N1}_{CV}_{MoM}_{item}.png")

    # Show the plot
    # plt.show()

def fig_distribution(data,N1,CV,MoM,item):
    # Create a histogram
    fig, ax = plt.subplots()
    plt.hist(data, bins=20, color='blue', alpha=0.7)
    plt.title('Random Number Distribution_{N1}_{CV}_{MoM}_{item}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    fig.savefig(f"Random Number Distribution_{N1}_{CV}_{MoM}_{item}.png")
    # Display the histogram
    # plt.show()

# Define the directory where your text files are located
directory = "C:/Users/User/MC_SIM_2SD/GPM_MC_2SD_higher_orders_2_compare_20230916"
# directory = "C:/Users/User/MC-sim/GPM_MC_nMonte_100000_MoM_compare"

# Get a list of files that match the pattern in the directory
matching_files = [file for file in os.listdir(directory) if file.startswith("GPM_MC_nMonte_100000_N") and file.endswith(".csv")]

rSampleOfRandoms = []

# Regular expression pattern to match N1 and CV
pattern = r'N_(\d+)_CV_([\d.]+)'

# col_list = [
#     'ln_ratio',
#     'se_ln_ratio',
#     'rSampleMeanLogScale1',
#     'rSampleSDLogScale1',
#     'rSampleMeanLogScale2',
#     'rSampleSDLogScale2',

# ]

col_list2 = [

    'ln_ratio',

]
    # 'percentile_2_5',
    # 'percentile_97_5',
    # 'se_ln_ratio',
    # 'P_value'
N1_CV = []
# Extract N1 and CV numbers
for file_name in matching_files:
    match = re.search(pattern, file_name)
    filename_frag = file_name.rstrip('.csv').split('_')
    MethodOfMoments = "_".join(filename_frag[9:])
    
    if match:
        N1 = int(match.group(1))
        CV = float(match.group(2))
        N1_CV.append((N1, CV))

        file_path = os.path.join(directory, file_name)
        df = pd.read_csv(file_path)
        for col in col_list2:
            data = df[col]
            # normal_quantile_plot(data,N1,CV,MethodOfMoments,str(col))
            prob_plot(data,N1,CV,MethodOfMoments,str(col))
            # fig_distribution(data,N1,CV,MethodOfMoments,str(col))


quit()
