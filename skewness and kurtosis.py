import numpy as np
from scipy.stats import kurtosis, skew, kurtosistest
from scipy.stats import kstest, uniform, norm
import numpy as np
import pandas as pd
from determine_type_I_error_module import SimulPivotMC
from datetime import datetime
import os

# Define the directory where your text files are located
directory = "C:/Users/User/MC_SIM_2SD/GPM_MC_2SD_higher_orders_2_compare_20230916"
# directory = "C:/Users/User/MC-sim/GPM_MC_nMonte_100000_MoM_compare"

# Get a list of files that match the pattern in the directory
matching_files = [file for file in os.listdir(directory) if file.startswith("GPM_MC_nMonte_100000_N") and file.endswith(".csv")]
# matching_files = ['GPM_MC_nMonte_100000_N_15_CV_0.5_20230916034310_higher_orders_of_moments_mean.csv']


data = {'MethodOfMoments':[], 'nMonte': [], 'N': [], 'CV': [], 'parameters':[], 'p_value':[], 'skewness':[], 'kurtosis':[], 'kurtosistest_p_value': []}
# Loop through the matching files and extract data
for filename in matching_files:

    filename_frag = filename.rstrip('.csv').split('_')
    MethodOfMoments = '_'.join(filename_frag[9:])
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
    Sries_ln_ratio_uniform.dropna(inplace=True)
    statistic, p_value = kstest(Sries_ln_ratio_uniform, 'norm')
        
    # Set your significance level (alpha)
    alpha = 0.05

    if p_value > alpha:
        print(f"Kolmogorov_nMonte_{nMonte}_N_{N}_CV_{CV}_{MethodOfMoments} Data appears to be normally distributed (fail to reject H0)")
        output_txt = f"Kolmogorov_nMonte_{nMonte}_N_{N}_CV_{CV}_{MethodOfMoments} Data appears to be normally distributed (fail to reject H0)"
    else: # p_value < alpha
        print(f"Kolmogorov_nMonte_{nMonte}_N_{N}_CV_{CV}_{MethodOfMoments} Data does not appear to be normally distributed (reject H0)")
        output_txt = f"Kolmogorov_nMonte_{nMonte}_N_{N}_CV_{CV}_{MethodOfMoments} Data does not appear to be normally distributed (reject H0)"

    data['parameters'].append(output_txt)
    data['p_value'].append(p_value)
    data['skewness'].append(skew(Sries_ln_ratio_uniform))
    data['kurtosis'].append(kurtosis(Sries_ln_ratio_uniform))
    res = kurtosistest(Sries_ln_ratio_uniform)
    data['kurtosistest_p_value'].append(res.pvalue)

end_time = datetime.now() #            
output_dir = f"Normality_test_nMonte_ln_ratio_{str(end_time).split('.')[0].replace('-','').replace(' ','').replace(':','')}"
print('save to', output_dir)
pd.DataFrame(data).to_csv(output_dir + '.csv')
quit()
#Data appears to be uniformly distributed (fail to reject H0)

# Generate a sample from a normal distribution
np.random.seed(0)  # for reproducibility
data = np.random.normal(0, 1, 1000)  # mean=0, standard deviation=1

# Calculate skewness
skewness = skew(data)

print("Skewness of the data:", skewness)



# Generate a sample from a normal distribution
np.random.seed(0)  # for reproducibility
data = np.random.normal(0, 1, 1000)  # mean=0, standard deviation=1

# Calculate excess kurtosis
excess_kurtosis = kurtosis(data)

print("Excess Kurtosis of the data:", excess_kurtosis)

