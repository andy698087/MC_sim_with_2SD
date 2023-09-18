from scipy.stats import kstest, uniform
import numpy as np
import pandas as pd
from determine_type_I_error_module import SimulPivotMC
from datetime import datetime

# Define the directory where your text files are located
directory = "C:/Users/User/MC-sim/GPM_MC_nMonte_100000_MoM_compare"

# Get a list of files that match the pattern in the directory
matching_files = [file for file in os.listdir(directory) if file.startswith("GPM_MC_nMonte_100000_N1") and file.endswith(".csv")]



data = {'MethodOfMoments':[], 'nMonte': [], 'N': [], 'CV': [], 'nSample':[], 'mean_intervals_include_zero':[], 'mean_weighted_ave_ln_ratio': [], 'se_weighted_ave_ln_ratio':[], 'mean_weighted_se_ln_ratio':[], 'se_weighted_se_ln_ratio':[]}

# Loop through the matching files and extract data
for filename in matching_files:
    filename_frag = filename.rstrip('.csv').split('_')
    data['MethodOfMoments'].append(filename_frag[-1])

    # Extract relevant information from each file
    df = pd.read_csv(os.path.join(directory,filename))

    data['nMonte'].append(filename_frag[3])
    data['N'].append(filename_frag[5])
    data['CV'].append(filename_frag[7])
    data['nSample'].append(nSample)
    nSim = 10000 #int(len(df)/nSample) 

    liest_ln_ratio_uniform = df['rSampleOfRandoms']
    c = 0
    # Assuming your data is in a NumPy array called 'data'
    statistic, p_value = kstest(liest_ln_ratio_uniform, cdf='uniform', args=(liest_ln_ratio_uniform.min(), liest_ln_ratio_uniform.max()))
    if c == 0:
        min_sample = np.min(liest_ln_ratio_uniform)
        max_sample = np.max(liest_ln_ratio_uniform)
    if c >= 0:
        min_sample_diff = min_sample - np.min(liest_ln_ratio_uniform)
        min_sample = np.min(liest_ln_ratio_uniform)
        max_sample_diff = max_sample - np.max(liest_ln_ratio_uniform)
        max_sample = np.max(rSampleOfRandoms_uniform)            
    c += liest_ln_ratio_uniform
        
    # Set your significance level (alpha)
    alpha = 0.05

    if p_value > alpha:
        print(f"Kolmogorov_nMonte_{nMonte}_N_{N}_CV_{CV} Data appears to be uniformly distributed (fail to reject H0)")
    else: # p_value < alpha
        print(f"Kolmogorov_nMonte_{nMonte}_N_{N}_CV_{CV} Data does not appear to be uniformly distributed (reject H0)")
    
    output_txt = f"Kolmogorov_nMonte_{nMonte}_N1_{N}_CV_{CV}"

    dict_result['parameters'].append(output_txt)
    dict_result['p_value'].append(p_value)
    dict_result['max_sample_diff'].append(max_sample_diff)
    dict_result['min_sample_diff'].append(min_sample_diff)
    dict_result['max_sample'].append(max_sample)
    dict_result['min_sample'].append(min_sample)
    

end_time = datetime.now() #            
output_dir = f"Uniformity_test_nMonte_{nMonte}_N_{N}_CV_{CV}_le_ratio_{str(end_time).split('.')[0].replace('-','').replace(' ','').replace(':','')}"
print('save to', output_dir)
pd.DataFrame(data).to_csv(output_dir + '.csv')
quit()
#Data appears to be uniformly distributed (fail to reject H0)