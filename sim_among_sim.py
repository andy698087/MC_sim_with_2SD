
import pandas as pd
import os
import re
import dask.dataframe as dd
from numpy.random import rand, seed
from numpy import sum, sqrt, mean, std
from scipy.stats import norm
from datetime import datetime


def making_random_pick(seed_unique):
    seed(seed_unique)
    df_rand = pd.DataFrame({'index_':range(100000),'rand':rand(100000)})
    return df_rand.sort_values(by='rand',ascending=True).iloc[:10,0].tolist()

def sample_weighted_ave_se_interval_include(list_choose_no):
    sum_wi_Ti, sum_wi_ = 0, 0
    for i in list_choose_no:
        sum_wi_Ti += df['ln_ratio'][i] * (1/(df['se_ln_ratio'][i]**2))
        sum_wi_ += 1/(df['se_ln_ratio'][i]**2)
    weighted_ave = sum_wi_Ti/sum_wi_
    weighted_se = sqrt(1/sum_wi_)
    upper_95CI = weighted_ave + z_score * weighted_se
    lower_95CI = weighted_ave - z_score * weighted_se
    intervals_include_zero = int((lower_95CI <= 0) and (upper_95CI >= 0))
    return weighted_ave, weighted_se, intervals_include_zero


start_time = datetime.now() # record the datetime at the start
print('start_time:', start_time) # print the datetime at the start

# Define the directory where your text files are located
directory = "C:/Users/User/MC-sim/GPM_MC_nMonte_100000_MoM_compare"

# Get a list of files that match the pattern in the directory
matching_files = [file for file in os.listdir(directory) if file.startswith("GPM_MC_nMonte_100000_N1") and file.endswith(".csv")]

# Initialize an empty list to store data from all files

data = {'MethodOfMoments':[], 'nMonte': [], 'N': [], 'CV': [], 'nSample':[], 'mean_intervals_include_zero':[], 'mean_weighted_ave_ln_ratio': [], 'se_weighted_ave_ln_ratio':[], 'mean_weighted_se_ln_ratio':[], 'se_weighted_se_ln_ratio':[]}

alpha = 0.05
z_score = norm.ppf(1 - alpha / 2) # two-tailed

list_nSamples = [5,10,20]
seed_ = 20230908

# Loop through the matching files and extract data
for filename in matching_files:
    filename_frag = filename.rstrip('.csv').split('_')
    if filename_frag[-1] == 'noMethodOfMoments' or filename_frag[-1] == 'False':
        continue
        data['MethodOfMoments'].append('No_MethodOfMoments')
    elif filename_frag[-1] == 'Higgins1':
        # data['MethodOfMoments'].append('Higgins1')
        print('Higgins1')
    elif filename_frag[-1] == 'Higgins2':
        continue
        data['MethodOfMoments'].append('Higgins2')
    else:
        # data['MethodOfMoments'].append('Orignal_MethodOfMoments')
        print('Orignal_MethodOfMoments')
    
    if filename_frag[5] == 100 or filename_frag[5] == 2: 
        continue
    
    

    # Extract relevant information from each file
    df = pd.read_csv(os.path.join(directory,filename))
    for nSample in list_nSamples:
        if filename_frag[-1] == 'Higgins1':
            data['MethodOfMoments'].append('Higgins1')
        else:
            data['MethodOfMoments'].append('Orignal_MethodOfMoments')
        
        data['nMonte'].append(filename_frag[3])
        data['N'].append(filename_frag[5])
        data['CV'].append(filename_frag[7])
        data['nSample'].append(nSample)
        nSim = int(len(df)/nSample) #10000

        df_for_dd = pd.DataFrame({'seed_':range(seed_, seed_ + nSim)})
        df_for_dd = df_for_dd['seed_'].apply(making_random_pick)
        ddf = dd.from_pandas(df_for_dd, npartitions=30)
        ddf= ddf.apply(sample_weighted_ave_se_interval_include, meta=('float64', 'float64'))
        df_record = pd.DataFrame(ddf.compute().tolist(), columns=['weighted_ave', 'weighted_se', 'intervals_include_zero'])

        data['mean_intervals_include_zero'].append(df_record['intervals_include_zero'].mean())
        data['mean_weighted_ave_ln_ratio'].append(df_record['weighted_ave'].mean())
        data['se_weighted_ave_ln_ratio'].append(df_record['weighted_ave'].std()/nSim)     
        data['mean_weighted_se_ln_ratio'].append(df_record['weighted_se'].mean())
        data['se_weighted_se_ln_ratio'].append(df_record['weighted_se'].std()/nSim)

# Create a DataFrame from the list of data
df = pd.DataFrame(data)
df[df.columns[1:]] = df[df.columns[1:]].astype(float)
# # Define the filename for your Excel file
df.to_csv("From_text_GPM_MC_nMonte_100000_N_Higgins1MoM_resample_mean_se.csv")

end_time = datetime.now() # record the datetime at the end
print('end_time:', end_time) # print the datetime at the end
time_difference = end_time - start_time
print('time_difference:', time_difference) # calculate the time taken
quit()