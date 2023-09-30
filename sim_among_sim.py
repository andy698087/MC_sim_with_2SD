
import pandas as pd
import os
import re
import dask.dataframe as dd
from numpy.random import rand, seed
from numpy import sum, sqrt, mean, std
from scipy.stats import norm
from datetime import datetime


def making_random_pick(row, nSample):
    # print(f'seed: {row}')
    seed(row)
    df_rand = pd.DataFrame({'index_': range(100000), 'rand': rand(100000)})
    # print(f'df_rand:{df_rand}')
    return df_rand.sort_values(by='rand', ascending=True).iloc[:nSample, 0].tolist()

def making_random_pick_no_replacement(seed_unique):
    # print(f'seed:{seed_unique}')
    seed(seed_unique)
    df_rand = pd.DataFrame({'index_':range(100000),'rand':rand(100000)})
    # print(f'df_rand:{df_rand}')
    return df_rand.sort_values(by='rand',ascending=True)['index_'].tolist()

def sample_weighted_ave_se_interval_include(list_choose_no):
    sum_wi_Ti, sum_wi_ = 0, 0
    for i in list_choose_no:
        sum_wi_Ti += df['ln_ratio'][i] * (1/(df['se_ln_ratio'][i]**2))
        sum_wi_ += 1/(df['se_ln_ratio'][i]**2)
    # print(df['ln_ratio'][list_choose_no])
    weighted_ave = sum_wi_Ti/sum_wi_
    weighted_se = sqrt(1/sum_wi_)
    upper_95CI = weighted_ave + z_score * weighted_se
    lower_95CI = weighted_ave - z_score * weighted_se
    intervals_include_zero = int((lower_95CI < 0) and (upper_95CI > 0))
    # print(f'weighted_ave: {weighted_ave}')
    return weighted_ave, weighted_se, intervals_include_zero

def sample_ave_se_interval_include(list_choose_no):
    # print(df['ln_ratio'][list_choose_no])
    sampled_ave_ln_ratio = df['ln_ratio'][list_choose_no].mean()
    sampled_ave_se_ln_ratio = df['se_ln_ratio'][list_choose_no].mean()
    upper_95CI = sampled_ave_ln_ratio + z_score * sampled_ave_se_ln_ratio
    lower_95CI = sampled_ave_ln_ratio - z_score * sampled_ave_se_ln_ratio
    intervals_include_zero = int((lower_95CI < 0) and (upper_95CI > 0))
    # print(f'sampled_ave_ln_ratio: {sampled_ave_ln_ratio}')
    return sampled_ave_ln_ratio, sampled_ave_se_ln_ratio, intervals_include_zero

def save_raw_ln_ratio(list_choose_no):
    # print(f"df['ln_ratio']:\n{df['ln_ratio']}")
    # print(f'list_choose_no:\n{list_choose_no}')
    # print(f"df['ln_ratio'].iloc[list_choose_no]:\n{df['ln_ratio'].iloc[list_choose_no]}")
    # quit()
    # print(f"sampled_ave_ln_ratio = df['ln_ratio'][list_choose_no].mean(): {df['ln_ratio'][list_choose_no].mean()}")
    # print(f"sampled_ave_ln_ratio = df['ln_ratio'][list_choose_no].values.mean(): {df['ln_ratio'][list_choose_no].values.mean()}")
    return df['ln_ratio'].iloc[list_choose_no].tolist(), df['se_ln_ratio'].iloc[list_choose_no].tolist()

start_time = datetime.now() # record the datetime at the start
print('start_time:', start_time) # print the datetime at the start

# Define the directory where your text files are located
directory = "C:/Users/User/MC_sim_2SD/Weibull_20230929_two_moments_mean_025_1_3"
directory = "/Users/andypchen/MC_sim_2SD/MC_sim_with_2SD/Weibull_20230929_two_moments_mean_025_1_3"

# Get a list of files that match the pattern in the directory
# matching_files = [file for file in os.listdir(directory) if file.startswith("Weibull_GPM_MC_nMonte_100000_N_") and file.endswith(".csv")]
# matching_files = [file for file in os.listdir(directory) if file.startswith("GPM_MC_nMonte_100000_N_15_CV_0.3_20230916192959_first_two_moment") and file.endswith(".csv")]
pattern = r"Weibull_GPMMC_nMonte_(\d+)_N_(\d+)_CV_(\d\.\d+)_Mean_([\d\.]+)_(\d{8}\d{6})_(\w+).csv"

# Get a list of files that match the pattern in the directory
matching_files = [file for file in os.listdir(directory) if re.match(pattern, file)]

weighted = False

if weighted:
    # Initialize an empty list to store data from all files
    data = {'MethodOfMoments':[], 'nMonte': [], 'N': [], 'CV': [], 'MeanTimeScale': [], 
            'nSample':[], 'nSim':[], 'replacement': [],
            'mean_intervals_include_zero':[], 'mean_weighted_ave_ln_ratio': [], 'se_weighted_ave_ln_ratio':[], 
            'mean_weighted_se_ln_ratio':[], 'se_weighted_se_ln_ratio':[]}
else:
    data = {'MethodOfMoments':[], 'nMonte': [], 'N': [], 'CV': [], 'MeanTimeScale': [], 
            'nSample':[], 'nSim':[], 'replacement': [],
            'mean_intervals_include_zero':[], 'mean_sampled_ave_ln_ratio': [], 'se_sampled_ave_ln_ratio':[], 
            'mean_sampled_se_ln_ratio':[], 'se_sampled_se_ln_ratio':[]}
    
alpha = 0.05
z_score = norm.ppf(1 - alpha / 2) # two-tailed

list_nSamples = [10]
seed_ = 20230908
        
for replacement in [True, False]:
    # Loop through the matching files and extract data
    for filename in matching_files:
        print(filename)

        # Extract relevant information from each file
        df = pd.read_csv(os.path.join(directory,filename))
        for nSample in list_nSamples:
            print(f'nSample: {nSample}')
            match = re.match(pattern, filename)
            if match.group(4) != '1':
                print('skip')
                continue

            data["nMonte"].append(int(match.group(1)))
            data["N"].append(int(match.group(2)))
            data["CV"].append(float(match.group(3)))
            data["MeanTimeScale"].append(match.group(4))
            data["MethodOfMoments"].append(match.group(6))
            
            data['nSample'].append(nSample)
            
            nSim = 10000 #int(len(df)/nSample) #10000
            data['nSim'].append(nSim)
            
            if replacement:
                print(f'relacement {replacement}')
                data['replacement'].append(replacement)
                # with replacement
                df_for_dd = pd.DataFrame({'seed_':range(seed_, seed_ + nSim)})
                df_record = df_for_dd
                df_for_dd = df_for_dd['seed_'].apply(making_random_pick, args=(nSample, ))
                df_record['rand_index'] = df_for_dd
                temp_results = df_for_dd.apply(save_raw_ln_ratio)
                pooled_ln_ratio = pd.DataFrame(temp_results.apply(lambda x: x[0]).tolist(), columns=[f"pooled_ln_ratio_{i+1}" for i in range(10)])
                pooled_se_ln_ratio = pd.DataFrame(temp_results.apply(lambda x: x[1]).tolist(), columns=[f"pooled_se_ln_ratio_{i+1}" for i in range(10)])
                df_record = pd.concat([df_record,pooled_ln_ratio],axis=1)
                df_record = pd.concat([df_record,pooled_se_ln_ratio],axis=1)
                del temp_results, pooled_ln_ratio, pooled_se_ln_ratio
                ddf = dd.from_pandas(df_for_dd, npartitions=30)
                # print(f'df_for_dd: {df_for_dd}')
                # print(f"len: {len(df_for_dd.iloc[0])}")
                # continue

            else:   
                print(f'relacement {replacement}')
                data['replacement'].append(replacement) 
                # no replacement
                rand_index = making_random_pick_no_replacement(seed_)
                sublist = [rand_index[i:i+nSample] for i in range(0, nSample*nSim, 10)]
                # print(f'sublist:\n{sublist}')
                # print(f'len sublist: {len(sublist)}, len sublist 0: {len(sublist[0])}')
                df_for_dd = pd.DataFrame({'rand_index':sublist})
                df_record = df_for_dd
                # df_record[['pooled_ln_ratio','pooled_se_ln_ratio']] = df_for_dd['rand_index'].apply(save_raw_ln_ratio)
                temp_results = df_for_dd['rand_index'].apply(save_raw_ln_ratio)
                pooled_ln_ratio = pd.DataFrame(temp_results.apply(lambda x: x[0]).tolist(), columns=[f"pooled_ln_ratio_{i+1}" for i in range(10)])
                pooled_se_ln_ratio = pd.DataFrame(temp_results.apply(lambda x: x[1]).tolist(), columns=[f"pooled_se_ln_ratio_{i+1}" for i in range(10)])
                df_record = pd.concat([df_record,pooled_ln_ratio],axis=1)
                df_record = pd.concat([df_record,pooled_se_ln_ratio],axis=1)
                del temp_results, pooled_ln_ratio, pooled_se_ln_ratio
                # print(f'df_record:\n {df_record}')
                ddf = dd.from_pandas(df_for_dd['rand_index'], npartitions=30)
                # print(f'df_for_dd: {df_for_dd}')
                # print(f"len: {len(df_for_dd.iloc[0])}")
                # print(f'df_for_dd.iloc[0]: {df_for_dd.iloc[0]}')
                # print('sample weighted', df_for_dd.iloc[0].apply(sample_weighted_ave_se_interval_include))
                # continue

            if weighted:
                print('weighted')
                ddf= ddf.apply(sample_weighted_ave_se_interval_include, meta=('float64', 'float64'))
                df_record.loc[:,['weighted_ave', 'weighted_se', 'intervals_include_zero']] = ddf.compute().tolist()
                
                data['mean_intervals_include_zero'].append(df_record['intervals_include_zero'].mean())
                data['mean_weighted_ave_ln_ratio'].append(df_record['weighted_ave'].mean())
                data['se_weighted_ave_ln_ratio'].append(df_record['weighted_ave'].std()/sqrt(nSim))     
                data['mean_weighted_se_ln_ratio'].append(df_record['weighted_se'].mean())
                data['se_weighted_se_ln_ratio'].append(df_record['weighted_se'].std()/sqrt(nSim))
                
            else:
                print('not weighted')
                ddf= ddf.apply(sample_ave_se_interval_include, meta=('float64', 'float64'))
                # print(ddf.compute().tolist())
                df_record[['sampled_ave_ln_ratio', 'sampled_ave_se_ln_ratio', 'intervals_include_zero']] = ddf.compute().tolist()
                # print(f'df_record:\n {df_record}')
                data['mean_intervals_include_zero'].append(df_record['intervals_include_zero'].mean())
                data['mean_sampled_ave_ln_ratio'].append(df_record['sampled_ave_ln_ratio'].mean())
                data['se_sampled_ave_ln_ratio'].append(df_record['sampled_ave_ln_ratio'].std()/sqrt(nSim))     
                data['mean_sampled_se_ln_ratio'].append(df_record['sampled_ave_se_ln_ratio'].mean())
                data['se_sampled_se_ln_ratio'].append(df_record['sampled_ave_se_ln_ratio'].std()/sqrt(nSim))
                # print(data)
            save_path = f"Sampled_mean_ave_ln_ratio_dask_{str(datetime.now()).split('.')[0].replace('-','').replace(' ','').replace(':','')}.csv"
            df_record.to_csv(f'Record_replacement_{replacement}_{save_path}')

# Create a DataFrame from the list of data
df = pd.DataFrame(data)
# df[df.columns[1:]] = df[df.columns[1:]].astype(float)
# # Define the filename for your Excel file

end_time = datetime.now() # record the datetime at the end
print('end_time:', end_time) # print the datetime at the end
save_path = f"Sampled_mean_ave_ln_ratio_dask_{str(end_time).split('.')[0].replace('-','').replace(' ','').replace(':','')}.csv"
df.to_csv(save_path)

print(f"save to {save_path}")
time_difference = end_time - start_time
print('time_difference:', time_difference) # calculate the time taken
quit()