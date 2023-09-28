
import pandas as pd
import os
import re
import dask.dataframe as dd
from numpy.random import rand, seed
from numpy import sum, sqrt, mean, std
from scipy.stats import norm
from datetime import datetime


def making_random_pick(row, nSample):
    seed(row)
    df_rand = pd.DataFrame({'index_': range(100000), 'rand': rand(100000)})
    return df_rand.sort_values(by='rand', ascending=True).iloc[:nSample, 0].tolist()

def making_random_pick_no_replacement(seed_unique):
    seed(seed_unique)
    df_rand = pd.DataFrame({'index_':range(100000),'rand':rand(100000)})
    return df_rand.sort_values(by='rand',ascending=True)['index_'].tolist()

def sample_weighted_ave_se_interval_include(list_choose_no):
    sum_wi_Ti, sum_wi_ = 0, 0
    for i in list_choose_no:
        sum_wi_Ti += df['ln_ratio'][i] * (1/(df['se_ln_ratio'][i]**2))
        sum_wi_ += 1/(df['se_ln_ratio'][i]**2)
    weighted_ave = sum_wi_Ti/sum_wi_
    weighted_se = sqrt(1/sum_wi_)
    upper_95CI = weighted_ave + z_score * weighted_se
    lower_95CI = weighted_ave - z_score * weighted_se
    intervals_include_zero = int((lower_95CI < 0) and (upper_95CI > 0))
    return weighted_ave, weighted_se, intervals_include_zero


start_time = datetime.now() # record the datetime at the start
print('start_time:', start_time) # print the datetime at the start

# Define the directory where your text files are located
directory = "C:/Users/User/MC_sim_2SD/Weibull_GPM_MC_2SD_20230925_two_moments_no_label_no_moments_not_LogSample"
# directory = "C:/Users/User/MC_sim_2SD/GPM_MC_2SD_higher_orders_2_compare_20230916_LogNorm"

# Get a list of files that match the pattern in the directory
matching_files = [file for file in os.listdir(directory) if file.startswith("Weibull_GPM_MC_nMonte_100000_N_") and file.endswith(".csv")]
# matching_files = [file for file in os.listdir(directory) if file.startswith("GPM_MC_nMonte_100000_N_15_CV_0.3_20230916192959_first_two_moment") and file.endswith(".csv")]

# Initialize an empty list to store data from all files

data = {'MethodOfMoments':[], 'nMonte': [], 'N': [], 'CV': [], 'nSample':[], 'mean_intervals_include_zero':[], 'mean_weighted_ave_ln_ratio': [], 'se_weighted_ave_ln_ratio':[], 'mean_weighted_se_ln_ratio':[], 'se_weighted_se_ln_ratio':[]}

alpha = 0.05
z_score = norm.ppf(1 - alpha / 2) # two-tailed

list_nSamples = [10]
seed_ = 20230908

# Loop through the matching files and extract data
for filename in matching_files:
    print(filename)
    filename_frag = filename.rstrip('.csv').split('_')
    # if filename_frag[-1] == 'noMethodOfMoments' or filename_frag[-1] == 'False':
    #     continue
    #     data['MethodOfMoments'].append('No_MethodOfMoments')
    # elif filename_frag[-1] == 'Higgins1':
    #     # data['MethodOfMoments'].append('Higgins1')
    #     print('Higgins1')
    # elif filename_frag[-1] == 'Higgins2':
    #     continue
    #     data['MethodOfMoments'].append('Higgins2')
    # else:
    #     # data['MethodOfMoments'].append('Orignal_MethodOfMoments')
    #     print('Orignal_MethodOfMoments')
    
    # if filename_frag[5] == '100' or filename_frag[5] == '2': 
    #     continue

    # Extract relevant information from each file
    df = pd.read_csv(os.path.join(directory,filename))
    for nSample in list_nSamples:
        # if filename_frag[-1] == 'Higgins1':
        #     data['MethodOfMoments'].append('Higgins1')
        # else:
        #     data['MethodOfMoments'].append('Orignal_MethodOfMoments')
        if 'no' in filename_frag and 'moment' in filename_frag:
            data['MethodOfMoments'].append('no_moments')
            print('no_moments')
        else:
            data['MethodOfMoments'].append('first_two_moments')
            print('first_two_moments')
        data['nMonte'].append(filename_frag[4])
        data['N'].append(filename_frag[6])
        data['CV'].append(filename_frag[8])
        data['nSample'].append(nSample)
        nSim = 10000 #int(len(df)/nSample) #10000

        # with replacement
        # df_for_dd = pd.DataFrame({'seed_':range(seed_, seed_ + nSim)})
        # df_for_dd = df_for_dd['seed_'].apply(making_random_pick, args=(nSample, ))
        # ddf = dd.from_pandas(df_for_dd, npartitions=30)
                
        # no replacement
        rand_index = making_random_pick_no_replacement(seed_)
        sublist = [rand_index[i:i+10] for i in range(0, nSample*nSim, 10)]
        df_for_dd = pd.DataFrame({'rand_index':sublist})
        ddf = dd.from_pandas(df_for_dd['rand_index'], npartitions=30)

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

end_time = datetime.now() # record the datetime at the end
print('end_time:', end_time) # print the datetime at the end
df.to_csv(f"Resample_mean_weighted_ave_ln_ratio_dask_{str(end_time).split('.')[0].replace('-','').replace(' ','').replace(':','')}.csv")
print(f"save to Resample_mean_weighted_ave_ln_ratio_dask_{str(end_time).split('.')[0].replace('-','').replace(' ','').replace(':','')}.csv")
time_difference = end_time - start_time
print('time_difference:', time_difference) # calculate the time taken
quit()