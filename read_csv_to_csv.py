"""
start_time: 2023-09-03 17:12:31.515876
end_time: 2023-09-03 17:16:15.667721
time_difference: 0:03:44.151845

nMonte = 10000; N1 = 50; CV= 0.5

ln ratio SE include zero: 0.9174
Count (ratio) P < 0.05: 826 (0.082600)
Count (ratio) P < 0.01: 203 (0.020300)
Percentile 2.5 > 0 count (ratio): 416 (0.041600)
Percentile 97.5 > 0 count (ratio): 440 (0.044000)
"""

import pandas as pd
import os
import re
from math import sqrt

def SE(data, N):
    return data/sqrt(N)

# Define a regular expression pattern to match content within parentheses
pattern = r'\((.*?)\)'

# Define the directory where your text files are located
directory = "C:/Users/User/MC-sim/GPM_MC_nMonte_100000_MoM_compare"

# Get a list of files that match the pattern in the directory
matching_files = [file for file in os.listdir(directory) if file.startswith("GPM_MC_nMonte_100000_N1") and file.endswith(".csv")]

# Initialize an empty list to store data from all files

data = {'MethodOfMoments':[], 'nMonte': [], 'N': [], 'CV': [], 'intervals_include_zero_mean': [], 'intervals_include_zero_SE': []}
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

    data['intervals_include_zero_mean'].append(df['intervals_include_zero'].mean())
    data['intervals_include_zero_SE'].append(df['intervals_include_zero'].std()/sqrt(len(df)))


    # Append the extracted data to the list

# Create a DataFrame from the list of data
df = pd.DataFrame(data)
df[df.columns[1:]] = df[df.columns[1:]].astype(float)
# # Define the filename for your Excel file
df.to_csv("From_text_GPM_MC_nMonte_100000_N_Higgins12MoMNo_SE.csv")

quit()