from scipy.stats import kstest, uniform
import numpy as np
import pandas as pd
from determine_type_I_error_module import SimulPivotMC
from datetime import datetime

nMonte = 5 # nMonte
dict_result = {'parameters':[],'p_value':[],'min_sample_diff':[],'max_sample_diff':[], 'min_sample': [], 'max_sample': []}
for N in [15, 25, 50]:
    for CV in [0.15, 0.3, 0.5]:
        
        run = SimulPivotMC(nMonte, N, CV)  # Generate the class SimulPivotMC(), generate variables in the def __init__(self)
        Series_rSampleOfRandoms_uniform = run.main_uniformity()  # start main()
        c = 0
        for rSampleOfRandoms_uniform in Series_rSampleOfRandoms_uniform:
            
            rSampleOfRandoms_uniform = np.array(rSampleOfRandoms_uniform)
            # Assuming your data is in a NumPy array called 'data'
            statistic, p_value = kstest(rSampleOfRandoms_uniform, cdf='uniform', args=(rSampleOfRandoms_uniform.min(), rSampleOfRandoms_uniform.max()))
            if c == 0:
                min_sample = np.min(rSampleOfRandoms_uniform)
                max_sample = np.max(rSampleOfRandoms_uniform)
            if c >= 0:
                min_sample_diff = min_sample - np.min(rSampleOfRandoms_uniform)
                min_sample = np.min(rSampleOfRandoms_uniform)
                max_sample_diff = max_sample - np.max(rSampleOfRandoms_uniform)
                max_sample = np.max(rSampleOfRandoms_uniform)            
            c += 1
                
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
output_dir = f"Uniformity_test_nMonte_{nMonte}_N_{N}_CV_{CV}_{str(end_time).split('.')[0].replace('-','').replace(' ','').replace(':','')}"
print('save to', output_dir)
pd.DataFrame(dict_result).to_csv(output_dir + '.csv')
quit()
#Data appears to be uniformly distributed (fail to reject H0)