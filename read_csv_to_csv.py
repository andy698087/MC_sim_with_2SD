import pandas as pd
import os
import re
from math import sqrt, log, exp
import numpy as np
from scipy.stats import norm
from datetime import datetime
import ast

def string_to_list(s):
    # Remove square brackets and split string by spaces
    items = s[1:-1].split()
    
    # Convert split strings back to floats
    return [float(item) for item in items]

def transform_from_raw_to_log_mean_SD(MeanTimeScale, SDTimeScale, CV):
    CV = SDTimeScale/MeanTimeScale
    CVsq = CV**2
    #Mean in log scale, Equation 7
    MeanLogScale = log(MeanTimeScale/sqrt(CVsq + 1)) 
    #SD in log scale, Equation 8
    SDLogScale = sqrt(log((CVsq + 1)))
    return MeanLogScale, SDLogScale

def transform_from_log_to_raw_mean_SD(MeanLogScale, CV):
    CVsq = CV**2
    MeanTimeScale  = exp(MeanLogScale)*sqrt(CVsq + 1)
    SDTimeScale = MeanTimeScale * CV
    return MeanTimeScale, SDTimeScale

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
directory = "C:/Users/User/MC_sim_2SD/Weibull_20230929_two_moments_mean_025_1_3"
# directory = "Weibull_GPM_MC_2SD_20230925_two_moments_no_label_no_moments_not_LogSample"
# Files list
files_list = os.listdir(directory)

# Get a list of files that match the pattern in the directory
# matching_files = [file for file in os.listdir(directory) if file.startswith("Weibull_GPMMC_nMonte_100000_N") and file.endswith("ts.csv")]
# print(matching_files)
# quit()
pattern = r"Weibull_GPMMC_nMonte_(\d+)_N_(\d+)_CV_(\d\.\d+)_Mean_([\d\.]+)_(\d{8}\d{6})_(\w+).csv"
# pattern = r"Weibull_GPM_MC_nMonte_(\d+)_N_(\d+)_CV_(\d\.\d+)_(\d{8}\d{6})_."

# Get a list of files that match the pattern in the directory
matching_files = [file for file in os.listdir(directory) if re.match(pattern, file)]
print(matching_files)


# Initialize an empty list to store data from all files
data = {'MethodOfMoments':[], 'nMonte': [], 'N': [], 'CV': [], 'MeanTimeScale': [],
        'mean_intervals_include_zero':[], 'mean_ln_ratio': [], 
        'se_mean_ln_ratio':[], 'mean_se_ln_ratio': [], 'se_mean_se_ln_ratio':[],
        'percentage_ln_ratio_above_0.5':[],'percentage_ln_ratio_below_0.5':[], 'percentage_95_coverage':[],
        'rSampleMeanTimeScale_avg':[], 'rSampleSDTimeScale_avg':[]}
# Loop through the matching files and extract data
for filename in matching_files:
    print(filename)
    match = re.match(pattern, filename)
    
    data["nMonte"].append(int(match.group(1)))
    data["N"].append(int(match.group(2)))
    data["CV"].append(float(match.group(3)))
    data["MeanTimeScale"].append(match.group(4))
    data["MethodOfMoments"].append(match.group(5))
    # print(data)
    # quit()
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

    data['rSampleMeanTimeScale_avg'].append(df['rSampleOfRandomsWeibull'].apply(string_to_list).apply(np.mean).mean())
    data['rSampleSDTimeScale_avg'].append(df['rSampleOfRandomsWeibull'].apply(string_to_list).apply(np.var).mean())
    
    # data['percentage_99_coverage'].append(df.apply(Coverage,args=('ln_ratio','se_ln_ratio',0.01),axis=1).mean())

    # Append the extracted data to the list

# Create a DataFrame from the list of data
df = pd.DataFrame(data)

# df[df.columns[1:]] = df[df.columns[1:]].astype(float)
# # Define the filename for your Excel file
end_time = datetime.now()
save_path = f"Weibull_from_csv_GPM_MC_nMonte_100000_SEmean_20230929_mean_025_1_3_{str(end_time).split('.')[0].replace('-','').replace(' ','').replace(':','')}.csv" 
df.to_csv(save_path)
print(f"save to {save_path}")
quit()