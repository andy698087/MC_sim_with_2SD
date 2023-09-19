from scipy.stats import kstest, uniform
import numpy as np
import pandas as pd
from determine_type_I_error_module import SimulPivotMC
from datetime import datetime
import os

# Define the directory where your text files are located
directory = "C:/Users/User/MC_SIM_2SD/GPM_MC_2SD_higher_orders_2_compare_20230916"
directory = "C:/Users/User/MC-sim/GPM_MC_nMonte_100000_MoM_compare"

# Get a list of files that match the pattern in the directory
matching_files = [file for file in os.listdir(directory) if file.startswith("GPM_MC_nMonte_100000_N") and file.endswith(".csv")]



data = {'MethodOfMoments':[], 'nMonte': [], 'N': [], 'CV': [], 'parameters':[], 'p_value':[], 'max_sample_diff': [], 'min_sample_diff':[], 'max_sample':[], 'min_sample':[]}
c=0
# Loop through the matching files and extract data
for filename in matching_files:
    filename_frag = filename.rstrip('.csv').split('_')
    MethodOfMoments = filename_frag[9:]
    data['MethodOfMoments'].append(MethodOfMoments)

    # Extract relevant information from each file
    df = pd.read_csv(os.path.join(directory,filename))
    nMonte = filename_frag[3]
    N = filename_frag[5]
    CV = filename_frag[7]
    data['nMonte'].append(nMonte)
    data['N'].append(N)
    data['CV'].append(CV)


    Sries_ln_ratio_uniform = df['ln_ratio']
    # list_ln_ratio_uniform = pd.Series([i for i in list_ln_ratio_uniform.strip('[]').split(',')]).astype(float)
    # print(list_ln_ratio_uniform)
    # Assuming your data is in a NumPy array called 'data'
    statistic, p_value = kstest(Sries_ln_ratio_uniform, cdf='uniform', args=(Sries_ln_ratio_uniform.min(), Sries_ln_ratio_uniform.max()))
    if c == 0:
        min_sample = np.min(Sries_ln_ratio_uniform)
        max_sample = np.max(Sries_ln_ratio_uniform)
    if c >= 0:
        min_sample_diff = min_sample - np.min(Sries_ln_ratio_uniform)
        min_sample = np.min(Sries_ln_ratio_uniform)
        max_sample_diff = max_sample - np.max(Sries_ln_ratio_uniform)
        max_sample = np.max(Sries_ln_ratio_uniform)            
    c += 1
        
    # Set your significance level (alpha)
    alpha = 0.05

    if p_value > alpha:
        print(f"Kolmogorov_nMonte_{nMonte}_N_{N}_CV_{CV}_{MethodOfMoments} Data appears to be uniformly distributed (fail to reject H0)")
        output_txt = f"Kolmogorov_nMonte_{nMonte}_N_{N}_CV_{CV}_{MethodOfMoments} Data appears to be uniformly distributed (fail to reject H0)"
    else: # p_value < alpha
        print(f"Kolmogorov_nMonte_{nMonte}_N_{N}_CV_{CV}_{MethodOfMoments} Data does not appear to be uniformly distributed (reject H0)")
        output_txt = f"Kolmogorov_nMonte_{nMonte}_N_{N}_CV_{CV}_{MethodOfMoments} Data does not appear to be uniformly distributed (reject H0)"

    data['parameters'].append(output_txt)
    data['p_value'].append(p_value)
    data['max_sample_diff'].append(max_sample_diff)
    data['min_sample_diff'].append(min_sample_diff)
    data['max_sample'].append(max_sample)
    data['min_sample'].append(min_sample)
    

end_time = datetime.now() #            
output_dir = f"Uniformity_test_nMonte_ln_ratio_{str(end_time).split('.')[0].replace('-','').replace(' ','').replace(':','')}"
print('save to', output_dir)
pd.DataFrame(data).to_csv(output_dir + '.csv')
quit()
#Data appears to be uniformly distributed (fail to reject H0)