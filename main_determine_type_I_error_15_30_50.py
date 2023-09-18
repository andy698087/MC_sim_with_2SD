"""
CV (15%, 30%, 50%) and sample size (15, 25, 50) 
"""
from determine_type_I_error_module import SimulPivotMC
import pandas as pd
import numpy as np
from scipy.stats import norm #, chi2
from datetime import datetime
# from math import sqrt, log, exp
import dask.dataframe as dd
# from dask import delayed
from sys import argv

# if __name__ == '__main__': # just a declare for starting process below, __main__ is the primary process of python
nMonteSim = 1 # nMonte
for MoM in [False]: # 'Higgins1', 'Higgins2', 'noMethodOfMoments', any others to 'original MoM'
    for N in [2]:
        for CV in [0.15]: # Missed original MoM, N = 25, CV = 0.5


            start_time = datetime.now() # record the datetime at the start
            print('start_time:', start_time) # print the datetime at the start
            print(f"Start GPM_MC_nMonteSim_{nMonteSim}_N_{N}_CV_{CV}_{str(start_time).split('.')[0].replace('-','').replace(' ','').replace(':','')}_{MoM}")

            run = SimulPivotMC(nMonteSim, N, CV)  # Generate the class SimulPivotMC(), generate variables in the def __init__(self)
            coverage_by_ln_ratio, df_record, nMonte, N1, CV1 = run.main(method_of_moments = MoM)  # start main()
            end_time = datetime.now() # record the datetime at the end
            print('end_time:', end_time) # print the datetime at the end
            time_difference = end_time - start_time
            print('time_difference:', time_difference) # calculate the time taken

            df_record['P_value'] = norm.cdf(np.abs(df_record['ln_ratio'])/df_record['se_ln_ratio'])

            P_below_0_05_Count_Percent = ((df_record['P_value'] > (1 - 0.05/2) ).astype(int).sum(), (df_record['P_value'] > (1 - 0.05/2) ).astype(int).mean())
            P_below_0_01_Count_Percent = ((df_record['P_value'] > (1 - 0.01/2) ).astype(int).sum(), (df_record['P_value'] > (1 - 0.01/2) ).astype(int).mean())
            print('Count (ratio) P < 0.05: %d (%f)' %P_below_0_05_Count_Percent)
            print('Count (ratio) P < 0.01: %d (%f)' %P_below_0_01_Count_Percent)
            
            print('ln ratio SE include zero: %s' %(coverage_by_ln_ratio,)) # print out the percentage of coverage
            
            df_record['percentile_2_5 > 0'] = (df_record['percentile_2_5'] > 0).astype(int)
            df_record['percentile_97_5 < 0'] = (df_record['percentile_97_5'] < 0).astype(int)    
            percentile_2_5_below_0 = (df_record['percentile_2_5 > 0'].sum(), df_record['percentile_2_5 > 0'].mean())
            percentile_97_5_above_0 = (df_record['percentile_97_5 < 0'].sum(), df_record['percentile_97_5 < 0'].mean())
            
            print('Percentile 2.5 > 0 count (ratio): %d (%f)' %percentile_2_5_below_0) # print out the percentage of coverage
            print('Percentile 97.5 < 0 count (ratio): %d (%f)' %percentile_97_5_above_0) # print out the percentage of coverage
            
            output_txt1 = f"start_time: {start_time}\nend_time: {end_time}\ntime_difference: {time_difference}\n\nnMonte = {nMonte}; N1 = {N1}; CV= {CV1}\n\nln ratio SE include zero: {coverage_by_ln_ratio}\n"
            
            output_dir = f"GPM_MC_nMonte_{nMonte}_N1_{N1}_CV_{CV1}_{str(end_time).split('.')[0].replace('-','').replace(' ','').replace(':','')}"
            
            print('csv save to ' + output_dir + f'_{MoM}.csv')
            df_record.to_csv(output_dir + f'_{MoM}.csv')

            with open(output_dir + f'_{MoM}.txt', 'w') as f:
                f.write(output_txt1)
                f.write("Count (ratio) P < 0.05: %d (%f)\n" %P_below_0_05_Count_Percent)
                f.write("Count (ratio) P < 0.01: %d (%f)\n" %P_below_0_01_Count_Percent)
                f.write('Percentile 2.5 > 0 count (ratio): %d (%f)\n' %percentile_2_5_below_0)
                f.write('Percentile 97.5 > 0 count (ratio): %d (%f)' %percentile_97_5_above_0)

quit()