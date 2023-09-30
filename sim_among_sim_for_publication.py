#This code works for generating results for Table 4, simulations among the results from previous simulations to calculate mean and SE of weighted average ln ratio
import pandas as pd
from numpy.random import rand, seed
from scipy.stats import norm
from datetime import datetime

# Function to randomly pick nSample elements based on a seed
def making_random_pick(seed_i, nSample):
    seed(seed_i)
    df_rand = pd.DataFrame({'index_': range(100000), 'rand': rand(100000)})
    return df_rand.sort_values(by='rand', ascending=True).iloc[:nSample, 0].tolist()

def making_random_pick_no_replacement(seed_unique):
    seed(seed_unique)
    df_rand = pd.DataFrame({'index_':range(100000),'rand':rand(100000)})
    return df_rand.sort_values(by='rand',ascending=True)['index_'].tolist()

# Function to calculate the weighted average of a sample
def sample_weighted_ave_se_interval_include(list_choose_no, df_GPM_MC_ln_ratio_se):
    sum_wi_Ti, sum_wi_ = 0, 0
    for i in list_choose_no:
        sum_wi_Ti += df_GPM_MC_ln_ratio_se['ln_ratio'][i] * (1 / (df_GPM_MC_ln_ratio_se['se_ln_ratio'][i] ** 2))
        sum_wi_ += 1 / (df_GPM_MC_ln_ratio_se['se_ln_ratio'][i] ** 2)
    weighted_ave = sum_wi_Ti / sum_wi_
    return weighted_ave

# Record the datetime at the start
start_time = datetime.now()
# Print the datetime at the start
print('start_time:', start_time)

# Define the file_path where your csv files are located
# csv_file_path = "/Users/andypchen/Dropbox/MC_sim/GPM_MC_nMonte_100000_N_50_CV_0.15_20230916014632_higher_orders_of_moments_noRawSamples.csv"
csv_file_path = "GPM_MC_2SD_higher_orders_2_compare_20230916_LogNorm/GPM_MC_nMonte_100000_N_15_CV_0.3_20230916192959_first_two_moment.csv"
# Initialize an empty dictionary to store data from all files
data = {'mean_weighted_ave_ln_ratio': [], 'se_weighted_ave_ln_ratio': []}

alpha = 0.05
# Two-tailed z-score
z_score = norm.ppf(1 - alpha / 2)

# Number of Samples to be sampled each iteration
nSamples = 10
# Number of simulations
nSim = 10000

# Extract relevant information from csv file, including columns of ln_ratio and se_ln_ratio
df_GPM_MC_ln_ratio_se = pd.read_csv(csv_file_path)
print(df_GPM_MC_ln_ratio_se)
seed_ = 20230908
list_seed = [s for s in range(seed_, seed_ + nSim)]
list_weighted_ave_ln_ratio = []

# Loop through different seeds, calculate weighted averages, and record in list
# for seed_i in list_seed:
    # random_pick_nSamples = making_random_pick(seed_i, nSamples)
    # list_weighted_ave_ln_ratio.append(sample_weighted_ave_se_interval_include(random_pick_nSamples, df_GPM_MC_ln_ratio_se))

# no replacement
rand_index = making_random_pick_no_replacement(seed_)
random_pick_nSamples = [rand_index[i:i+10] for i in range(0, nSamples*nSim, 10)]
print(random_pick_nSamples)
for random_pick_nSample in random_pick_nSamples:
    list_weighted_ave_ln_ratio.append(sample_weighted_ave_se_interval_include(random_pick_nSample, df_GPM_MC_ln_ratio_se))
print(list_weighted_ave_ln_ratio)
# Convert the list to a pandas Series for calculating mean and standard deviation
list_weighted_ave_ln_ratio = pd.Series(list_weighted_ave_ln_ratio)
data['mean_weighted_ave_ln_ratio'].append(list_weighted_ave_ln_ratio.mean())
data['se_weighted_ave_ln_ratio'].append(list_weighted_ave_ln_ratio.std() / nSim)

# Create a DataFrame from the collected data, and transform data into float
df_data = pd.DataFrame(data).astype(float)

# Record the datetime at the end
end_time = datetime.now()
print('end_time:', end_time)
# Calculate the time taken
time_difference = end_time - start_time
print('time_difference:', time_difference)

# Define the output filename for the CSV file with timestamp
output_dir = f"Resample_NR_mean_weighted_ave_ln_ratio_{str(end_time).split('.')[0].replace('-', '').replace(' ', '').replace(':', '')}.csv"

# Save the DataFrame to a CSV file
df_data.to_csv(output_dir)

quit()
