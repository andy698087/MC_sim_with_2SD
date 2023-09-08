import pandas as pd
import os
import re
from math import sqrt

# Define the directory where your text files are located
directory = "C:/Users/User/MC-sim/GPM_MC_nMonte_100000_MoM_compare"

# Get a list of files that match the pattern in the directory
matching_files = [file for file in os.listdir(directory) if file.startswith("GPM_MC_nMonte_100000_N1") and file.endswith(".csv")]

# Initialize an empty list to store data from all files

data = {'MethodOfMoments':[], 'nMonte': [], 'N': [], 'CV': [], 'mean_intervals_include_zero':[], 'mean_ln_ratio': [], 'se_mean_ln_ratio':[], 'mean_se_ln_ratio': [], 'se_mean_se_ln_ratio':[]}
# Loop through the matching files and extract data
for filename in matching_files:
    filename_frag = filename.rstrip('.csv').split('_')
    if filename_frag[-1] == 'noMethodOfMoments' or filename_frag[-1] == 'False':
        data['MethodOfMoments'].append('No_MethodOfMoments')
    elif filename_frag[-1] == 'Higgins1':
        data['MethodOfMoments'].append('Higgins1')
    elif filename_frag[-1] == 'Higgins2':
        data['MethodOfMoments'].append('Higgins2')
    else:
        data['MethodOfMoments'].append('Orignal_MethodOfMoments')

    data['nMonte'].append(filename_frag[3])
    data['N'].append(filename_frag[5])
    data['CV'].append(filename_frag[7])

    # Extract relevant information from each file
    df = pd.read_csv(os.path.join(directory,filename))
    
    # print(df.columns.tolist())
    # 'Seeds', 'rSampleOfRandoms_1' ... 'rSampleOfRandoms_200', 'rSampleMeanLogScale1', 'rSampleSDLogScale1', 'rSampleMeanLogScale2', 'rSampleSDLogScale2', 'lower_bound_SE', 'upper_bound_SE', 
    # 'ln_ratio', 'se_ln_ratio', 'percentile_2_5', 'percentile_97_5', 'intervals_include_zero', 'P_value', 'percentile_2_5 > 0', 'percentile_97_5 < 0']  
    data['mean_intervals_include_zero'].append(df['intervals_include_zero'].mean())
    data['mean_ln_ratio'].append(df['ln_ratio'].mean())
    data['se_mean_ln_ratio'].append(df['ln_ratio'].std()/sqrt(len(df)))
    data['mean_se_ln_ratio'].append(df['se_ln_ratio'].mean())
    data['se_mean_se_ln_ratio'].append(df['se_ln_ratio'].std()/sqrt(len(df)))


    # Append the extracted data to the list

# Create a DataFrame from the list of data
df = pd.DataFrame(data)
df[df.columns[1:]] = df[df.columns[1:]].astype(float)
# # Define the filename for your Excel file
df.to_csv("From_text_GPM_MC_nMonte_100000_N_Higgins12MoMNo_SEmean.csv")

quit()