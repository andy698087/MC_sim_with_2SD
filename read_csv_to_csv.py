import pandas as pd
import os
import re
from math import sqrt
import numpy as np
from scipy.stats import norm

def Coverage(row, col_ln_ratio, col_se_ln_ratio, alpha = 0.05):
    # print(row)
    z_score = norm.ppf(1 - alpha / 2)

    ln_ratio = row[col_ln_ratio]
    se_ln_ratio = row[col_se_ln_ratio]

    # Calculate the confidence intervals with z_score of alpha = 0.05, Equation 6
    lower_bound = ln_ratio - z_score * se_ln_ratio
    upper_bound = ln_ratio + z_score * se_ln_ratio   

    intervals_include_zero = (lower_bound < 0) and (upper_bound > 0)
    # 1 as True, 0 as False, check coverage
    return int(intervals_include_zero)  

# Define the directory where your text files are located
directory = "C:/Users/User/MC_sim_2SD/Weibull_20230927-2_MeanLogScale_0"

# Get a list of files that match the pattern in the directory
matching_files = [file for file in os.listdir(directory) if file.startswith("Weibull_GPMMC_nMonte_100000_N") and file.endswith(".csv")]

# Initialize an empty list to store data from all files

data = {'MethodOfMoments':[], 'nMonte': [], 'N': [], 'CV': [], 
        'mean_intervals_include_zero':[], 'mean_ln_ratio': [], 
        'se_mean_ln_ratio':[], 'mean_se_ln_ratio': [], 'se_mean_se_ln_ratio':[],
        'percentage_ln_ratio_above_0.5':[],'percentage_ln_ratio_below_0.5':[], 'percentage_95_coverage':[], 'percentage_99_coverage':[]}
# Loop through the matching files and extract data
for filename in matching_files:
    print(filename)
    filename_frag = filename.rstrip('.csv').split('_')
    # if filename_frag[-1] == 'noMethodOfMoments' or filename_frag[-1] == 'False':
    #     data['MethodOfMoments'].append('No_MethodOfMoments')
    # elif filename_frag[-1] == 'Higgins1':
    #     data['MethodOfMoments'].append('Higgins1')
    # elif filename_frag[-1] == 'Higgins2':
    #     data['MethodOfMoments'].append('Higgins2')
    # else:
    #     data['MethodOfMoments'].append('Orignal_MethodOfMoments')
    data['MethodOfMoments'].append('_'.join(filename_frag[9:]))
    data['nMonte'].append(filename_frag[3])
    data['N'].append(filename_frag[5])
    data['CV'].append(filename_frag[7])

    # Extract relevant information from each file
    df = pd.read_csv(os.path.join(directory,filename))
    # print(df)
    # print(df.columns.tolist())
    # 'Seeds', 'rSampleOfRandoms_1' ... 'rSampleOfRandoms_200', 'rSampleMeanLogScale1', 'rSampleSDLogScale1', 'rSampleMeanLogScale2', 'rSampleSDLogScale2', 'lower_bound_SE', 'upper_bound_SE', 
    # 'ln_ratio', 'se_ln_ratio', 'percentile_2_5', 'percentile_97_5', 'intervals_include_zero', 'P_value', 'percentile_2_5 > 0', 'percentile_97_5 < 0']  
    data['mean_intervals_include_zero'].append(df['intervals_include_zero'].mean())
    data['mean_ln_ratio'].append(df['ln_ratio'].mean())
    data['se_mean_ln_ratio'].append(df['ln_ratio'].std()/sqrt(len(df)))
    data['mean_se_ln_ratio'].append(df['se_ln_ratio'].mean())
    data['se_mean_se_ln_ratio'].append(df['se_ln_ratio'].std()/sqrt(len(df)))
    data['percentage_ln_ratio_above_0.5'].append(sum((df['ln_ratio']>0.5).astype(int))/len(df))
    data['percentage_ln_ratio_below_0.5'].append(sum((df['ln_ratio']<-0.5).astype(int))/len(df))
    data['percentage_95_coverage'].append(df.apply(Coverage,args=('ln_ratio','se_ln_ratio',0.05),axis=1).mean())
    data['percentage_99_coverage'].append(df.apply(Coverage,args=('ln_ratio','se_ln_ratio',0.01),axis=1).mean())

    # Append the extracted data to the list

# Create a DataFrame from the list of data
df = pd.DataFrame(data)

df[df.columns[1:]] = df[df.columns[1:]].astype(float)
# # Define the filename for your Excel file
df.to_csv("Weibull_from_csv_GPM_MC_nMonte_100000_SEmean_20230927_MeanLogScale_0.csv")

quit()