#This one, using the multithread, runs quickly (about 10 min for each 100,000 simulations). It obtains results identical to at least 16 digits as the one using no multithread.
import pandas as pd
import numpy as np
from scipy.stats import norm, chi2
from datetime import datetime
from math import sqrt, log, exp
import dask.dataframe as dd
import os
import re

class SimulPivotMC(object):
    def __init__(self, nMonteSim, N, CV, rpy_out_files_path):
        self.rpy_out_files_path = rpy_out_files_path

        # number of Monte Carlo Simulation
        self.nMonte = nMonteSim

        # Calculate z-score for alpha = 0.05, ppf is the percent point function that is inverse of cumulative distribution function
        self.z_score = norm.ppf(1 - 0.05 / 2)

        # Sample size, we choose 15, 25, 50, notation "n" in the manuscript
        self.N1 = N
        self.N2 = self.N1

        # coefficient of variation, we choose 0.15, 0.3, 0.5
        self.CV1 = CV
        self.CV2 = self.CV1

        # Mean in log scale, notation "μ_i" in the manuscript
        self.rMeanLogScale1 = 0
        self.rMeanLogScale2 = self.rMeanLogScale1
        
        # Standard deviation in log scale, notation "σ_i" in the manuscript, Equation 1 in the manuscript
        self.rSDLogScale1 = sqrt(log(1 + self.CV1 ** 2)) 
        self.rSDLogScale2 = self.rSDLogScale1


        # the number for pivot, the notation "m" in the manuscript
        nSimulForPivot = 100000-1

        # choosing a seed
        self.seed_value = 12181988

        # Generate 4 set of random numbers each with specified seed, will be used for U1, U2, Z1, and Z2 later (Equation 3)
        np.random.seed(self.seed_value - 1)
        random_numbers1_1 = np.random.rand(nSimulForPivot)

        np.random.seed(self.seed_value - 2)
        random_numbers1_2 = np.random.rand(nSimulForPivot)

        np.random.seed(self.seed_value - 3)
        random_numbers2_1 = np.random.rand(nSimulForPivot)

        np.random.seed(self.seed_value - 4)
        random_numbers2_2 = np.random.rand(nSimulForPivot)

        # Generate random number for later used in calculating Ui and Zi in generalized pivotal method
        # group 1 pivot calculation, U_i and Z_i used in Equation 3
        self.U1 = chi2.ppf(random_numbers1_1, self.N1 - 1 )
        self.Z1 = norm.ppf(random_numbers2_1)

        # group 2 pivot calculation, U_i and Z_i used in Equation 3        
        self.U2 = chi2.ppf(random_numbers1_2, self.N2 - 1 )
        self.Z2 = norm.ppf(random_numbers2_2)
    
    # the main process, method of moments = ['no_moments', 'first_two_moment', 'higher_orders_of_moments']
    def main(self, method):
        # the pre-determined list of seeds, using number of nMonte
        # list_seeds = [i for i in range(self.seed_value, self.seed_value + self.nMonte)] 
        # # put the list of seeds into a table (a.k.a DataFrame) with one column named "Seeds"  
        # df = pd.DataFrame({'Seeds':list_seeds}) 
        
        #using no method of moments 
        if method == 'no_moments': 
            print(f'method:{method}')
            print('Samples_normal')
            # generate log-normal distributed numbers, using mean of rMeanLogScale and standard deviation of rSDLogScale
            df['rSampleOfRandoms'] = df.apply(self.Samples_normal, args=('Seeds',), axis=1) 

            print('dask')
            # put the table into dask, a progress that can parallel calculating each rows using multi-thread
            df = dd.from_pandas(df['rSampleOfRandoms'], npartitions=16) 
            print('Mean_SD')
            # calculate sample mean and SD using Mean_SD
            df = df.apply(self.Mean_SD, meta=meta) 

        # using method of moments to transform from raw to log mean and SD
        elif method in ['Luo_Wan', 'bc', 'qe', 'mln']:
            print(f'method:{method}')
            # # generate log-normal distributed numbers, using mean of rMeanLogScale and standard deviation of rSDLogScale
            # print('Samples_normal + exponential')
            # df['rSampleOfRandoms'] = df.apply(self.Samples_normal, args=('Seeds',), axis=1).apply(lambda x: [np.exp(item) for item in x])            
            
            # # using Equation 7 and 8
            # if method in ['Luo_Wan', 'bc', 'qe', 'mln']:
            #     print(f'get_estimated_Mean_SD_from_Samples with method = {method}')
            #     # calculate sample mean and SD using Mean_SD
            #     df[['rSampleMeanRaw1', 'rSampleSDRaw1', 'rSampleMeanRaw2', 'rSampleSDRaw2']] = df['rSampleOfRandoms'].apply(self.get_estimated_Mean_SD_from_Samples, args=(method,)).tolist()
            print(f'read_csv: {rpy_out_files_path}')
            df = pd.read_csv(self.rpy_out_files_path)

            df_record = df[['rSampleMeanRaw1', 'rSampleSDRaw1', 'rSampleMeanRaw2', 'rSampleSDRaw2']].copy()
            


            print('dask')
            df = dd.from_pandas(df[['rSampleMeanRaw1', 'rSampleSDRaw1', 'rSampleMeanRaw2', 'rSampleSDRaw2']], npartitions=35) 
            meta = ('float64', 'float64')

            print('first_two_moment')
            # transform sample mean and SD in log scale using estimated Mean and SD
            df = df.apply(self.first_two_moment, axis=1, args=(0,1,2,3), meta=('float64', 'float64')) 
                           
        else:
            print('no method in main')

        df_record[['rSampleMeanLogScale1', 'rSampleSDLogScale1', 'rSampleMeanLogScale2', 'rSampleSDLogScale2']] = df.compute().tolist()
        
        print('GPM_log_ratio_SD')
        # Equation 3 # generate 'ln_ratio' and 'se_ln_ratio' with sample mean and SD using GPM
        df_SD = df.apply(self.GPM_log_ratio_SD, args=(0,1,2,3), meta=meta)  
        df_record[['ln_ratio_SD', 'se_ln_ratio_SD']] = df_SD.compute().tolist()

        print('Coverage SD')
        # check coverage of each rows
        df_SD = df_SD.apply(self.Coverage, args=(0,1,0), meta=meta) 
        df_record['coverage_SD'] = df_SD.compute().tolist()        

        print('Mean coverage_SD')
        # compute the mean of the list of coverage (0 or 1), it equals to the percentage of coverage
        coverage_SD = df_record['coverage_SD'].mean()

        print('GPM_log_ratio_Mean')
        # Equation 3 # generate 'ln_ratio' and 'se_ln_ratio' with sample mean and SD using GPM
        df_Mean = df.apply(self.GPM_log_ratio_Mean, args=(0,1,2,3), meta=meta)  
        df_record[['ln_ratio_Mean', 'se_ln_ratio_Mean']] = df_Mean.compute().tolist()

        print('Coverage Mean')
        # check coverage of each rows
        df_Mean = df_Mean.apply(self.Coverage, args=(0,1,0), meta=meta) 
        df_record['coverage_Mean'] = df_Mean.compute().tolist()        

        print('Mean coverage_Mean')
        # compute the mean of the list of coverage (0 or 1), it equals to the percentage of coverage
        coverage_Mean = df_record['coverage_Mean'].mean()

        return coverage_SD, coverage_Mean, df_record, self.nMonte, self.N1, self.CV1, method

    
    def Samples_normal(self, row, seed_):        
        # # using seed from pre-determined list
        # # generate log-normal distribution, using mean of rMeanLogScale and standard deviation of rSDLogScale
        # rSampleOfRandoms = norm.rvs(loc=self.rMeanLogScale1, scale=self.rSDLogScale1, size = self.N1+self.N2, random_state = row[seed_])
       
       # using seed from pre-determined list
        np.random.seed(row[seed_]) 
        # generate log-normal distribution, using mean of rMeanLogScale and standard deviation of rSDLogScale
        rSampleOfRandoms = [(norm.ppf(i,loc=self.rMeanLogScale1, scale=self.rSDLogScale1)) for i in np.random.rand(self.N1+self.N2)] 
        
        return rSampleOfRandoms

    def Mean_SD(self, row):

        rSampleOfRandoms1 = row[:self.N1]
        rSampleOfRandoms2 = row[self.N1:(self.N1+self.N2)]
        
        # the mean of rSampleOfRandoms1, notation "z_i"
        rSampleMean1 = np.mean(rSampleOfRandoms1)  
        # the standard deviation of rSampleOfRandoms1, delta degree of freeden = 1, notation "sz_i"
        rSampleSD1 = np.std(rSampleOfRandoms1, ddof=1) 
        rSampleMean2 = np.mean(rSampleOfRandoms2)
        rSampleSD2 = np.std(rSampleOfRandoms2, ddof=1)

        return rSampleMean1, rSampleSD1, rSampleMean2, rSampleSD2

    def get_estimated_Mean_SD_from_Samples(self, row, method = 'Luo_Wan'):
        rSampleOfRandoms1 = row[:self.N1]
        rSampleOfRandoms2 = row[self.N1:(self.N1+self.N2)]

        q1_1 = pd.Series(rSampleOfRandoms1).quantile(.25)
        median_1 = pd.Series(rSampleOfRandoms1).quantile(.5)
        q3_1 = pd.Series(rSampleOfRandoms1).quantile(.75)

        q1_2 = pd.Series(rSampleOfRandoms2).quantile(.25)
        median_2 = pd.Series(rSampleOfRandoms2).quantile(.5)
        q3_2 = pd.Series(rSampleOfRandoms2).quantile(.75)
        
        if method == 'Luo_Wan':
            # the mean of rSampleOfRandoms1, notation "z_i"
            # the standard deviation of rSampleOfRandoms1, delta degree of freeden = 1, notation "sz_i"
            rSampleMean1, rSampleSD1 = self.ThreeValues_to_Mean_SD_Luo_Wan(q1_1, median_1, q3_1, self.N1)
            rSampleMean2, rSampleSD2 = self.ThreeValues_to_Mean_SD_Luo_Wan(q1_2, median_2, q3_2, self.N2)
        elif method in ['bc', 'qe', 'mln']:
            
            robjects.r(f"""
            library(estmeansd)
            set.seed(1)
            mean_sd1 <- {method}.mean.sd(q1.val = {q1_1}, med.val = {median_1}, q3.val = {q3_1}, n = {self.N1})
            mean_sd2 <- {method}.mean.sd(q1.val = {q1_2}, med.val = {median_2}, q3.val = {q3_2}, n = {self.N2})
                    """)

            rSampleMean1, rSampleSD1 = robjects.r['mean_sd1'][0][0], robjects.r['mean_sd1'][1][0]
            rSampleMean2, rSampleSD2 = robjects.r['mean_sd2'][0][0], robjects.r['mean_sd2'][1][0]

        else:
            print('not right method in get_estimated_Mean_SD_from_Samples')

        return rSampleMean1, rSampleSD1, rSampleMean2, rSampleSD2

    def ThreeValues_to_Mean_SD_Luo_Wan(self, q1, median, q3, N):
        opt_weight = 0.7 + 0.39/N
        # Luo 2018
        est_mean = opt_weight * (q1 + q3)/2 + (1-opt_weight) * median
        # Wan 2014
        est_SD = (q3 - q1)/(2 * norm.ppf((0.75 * N - 0.125)/(N + 0.25))) # loc=0, scale=1

        return est_mean, est_SD
    
    def first_two_moment(self, row, col_SampleMean1, col_SampleSD1, col_SampleMean2, col_SampleSD2):
        
        SampleMean1 = row.iloc[col_SampleMean1]
        SampleSD1 = row.iloc[col_SampleSD1]

        SampleMean2 = row.iloc[col_SampleMean2]
        SampleSD2 = row.iloc[col_SampleSD2]
        #using Equation 7 and 8
        rSampleMeanLogScale1, rSampleSDLogScale1 = self.transform_from_raw_to_log_mean_SD(SampleMean1, SampleSD1)
        rSampleMeanLogScale2, rSampleSDLogScale2 = self.transform_from_raw_to_log_mean_SD(SampleMean2, SampleSD2)
                   
        return rSampleMeanLogScale1, rSampleSDLogScale1, rSampleMeanLogScale2, rSampleSDLogScale2

    def transform_from_raw_to_log_mean_SD(self, Mean, SD):
        CV = SD/Mean
        CVsq = CV**2
        #Mean in log scale, Equation 7
        MeanLogScale_1 = log(Mean/sqrt(CVsq + 1)) 
        #SD in log scale, Equation 8
        SDLogScale_1 = sqrt(log((CVsq + 1)))
        #SD in log scale, Equation 9
        # SDLogScale_2 = sqrt(CVsq * (1 + CVsq/(1 + CVsq))**2 - (1 + CVsq/(1 + CVsq)) * ((1 + CVsq)**2 - 3 + 2/(1 + CVsq)) + (1/4) * ((1 + CVsq)**4 - 4*(1 + CVsq) - 1 + 8/(1 + CVsq) - 4/((1 + CVsq)**2)))   
        
        return MeanLogScale_1, SDLogScale_1
    
    def GPM_log_ratio_SD(self, row, col_SampleMeanLog1, col_SampleSDLog1, col_SampleMeanLog2, col_SampleSDLog2):  # Equation 2 and 3 

        #group 1 pivot calculation
        SampleMeanLog1 = row[col_SampleMeanLog1]
        SampleSDLog1 = row[col_SampleSDLog1]
        Pivot1 = self.Pivot_calculation_2SD(SampleMeanLog1, SampleSDLog1, self.N1, self.U1, self.Z1)
        #group 2 pivot calculation
        SampleMeanLog2 = row[col_SampleMeanLog2]
        SampleSDLog2 = row[col_SampleSDLog2]
        Pivot2 = self.Pivot_calculation_2SD(SampleMeanLog2, SampleSDLog2, self.N2, self.U2, self.Z2)

        # Equation 2, generalized pivotal statistics
        pivot_statistics = np.log(Pivot1) - np.log(Pivot2)

        # Calculate ln ratio and SE ln ratio by percentile and Z statistics, Equation 4 and 5
        ln_ratio = pd.Series(pivot_statistics).quantile(.5)
        se_ln_ratio = (pd.Series(pivot_statistics).quantile(.75) - pd.Series(pivot_statistics).quantile(.25))/(norm.ppf(.75) - norm.ppf(.25))
        
        return ln_ratio, se_ln_ratio
    
    def GPM_log_ratio_Mean(self, row, col_SampleMeanLog1, col_SampleSDLog1, col_SampleMeanLog2, col_SampleSDLog2):  # Equation 2 and 3 

        #group 1 pivot calculation
        SampleMeanLog1 = row[col_SampleMeanLog1]
        SampleSDLog1 = row[col_SampleSDLog1]
        Pivot1 = self.Pivot_calculation_2mean(SampleMeanLog1, SampleSDLog1, self.N1, self.U1, self.Z1)
        # print(f'Mean Pivot1:{Pivot1}')

        #group 2 pivot calculation
        SampleMeanLog2 = row[col_SampleMeanLog2]
        SampleSDLog2 = row[col_SampleSDLog2]
        Pivot2 = self.Pivot_calculation_2mean(SampleMeanLog2, SampleSDLog2, self.N2, self.U2, self.Z2)
        # print(f'Mean Pivot2:{Pivot2}')

        # Equation 2, generalized pivotal statistics
        pivot_statistics = Pivot1 - Pivot2 # ln(exp(Pivot1 - Pivot2))
        # print(f'Mean pivot_statistics:{pivot_statistics}')

        # print(f'mean 0.025p {pd.Series(pivot_statistics).quantile(.025)}, mean 0.975p {pd.Series(pivot_statistics).quantile(.975)}')
        # print(f'mean 0.025p {pd.Series(pivot_statistics).quantile(.025)}, mean 0.975p {pd.Series(pivot_statistics).quantile(.975)}')
        
        # Calculate ln ratio and SE ln ratio by percentile and Z statistics, Equation 4 and 5
        ln_ratio = pd.Series(pivot_statistics).quantile(.5)
        se_ln_ratio = (pd.Series(pivot_statistics).quantile(.75) - pd.Series(pivot_statistics).quantile(.25))/(norm.ppf(.75) - norm.ppf(.25))
        # # print(f'ln_ratio: {ln_ratio}, se_ln_ratio: {se_ln_ratio}')
        # print(f'lower_bound: { ln_ratio - self.z_score * se_ln_ratio}, upper_bound: {ln_ratio + self.z_score * se_ln_ratio   }')

        return ln_ratio, se_ln_ratio
        
        # lower_bound = pd.Series(pivot_statistics).quantile(.025)
        # upper_bound = pd.Series(pivot_statistics).quantile(.975)
        # intervals_include_zero = (lower_bound < 1) and (upper_bound > 1)
        # return int(intervals_include_zero)
    
    def Pivot_calculation_2SD(self, rSampleMeanLogScale, rSampleSDLogScale, N, U, Z):
        return np.exp(rSampleMeanLogScale- np.sqrt((rSampleSDLogScale**2 * (N-1))/U) * (Z/sqrt(N)) ) * np.sqrt((np.exp((rSampleSDLogScale**2 * (N-1))/U) - 1) * np.exp((rSampleSDLogScale**2 * (N-1))/U))

    def Pivot_calculation_2mean(self, rSampleMeanLogScale, rSampleSDLogScale, N, U, Z):
        sqrt_U = np.sqrt(U)
        return rSampleMeanLogScale - (Z/sqrt(N)) * (rSampleSDLogScale/(sqrt_U/sqrt(N-1))) + 0.5 * (rSampleSDLogScale/(sqrt_U/sqrt(N-1))) ** 2

    def Coverage(self, row, col_ln_ratio, col_se_ln_ratio, ideal):
        
        ln_ratio = row[col_ln_ratio]
        se_ln_ratio = row[col_se_ln_ratio]

        # Calculate the confidence intervals with z_score of alpha = 0.05, Equation 6
        lower_bound = ln_ratio - self.z_score * se_ln_ratio
        upper_bound = ln_ratio + self.z_score * se_ln_ratio   
        
        intervals_include_zero = (lower_bound < ideal) and (upper_bound > ideal)
        # 1 as True, 0 as False, check coverage
        return int(intervals_include_zero)  
    
if __name__ == '__main__':
    
    output_folder = "MeanSD_From3ValuesInRaw_BCQEMLN_rpy2out_20231104"
    files_list = os.listdir(output_folder)
    pattern = r"MeanSD_From5Values_nMonte_(\d+)_N_(\d+)_CV_(\d\.\d+)_rpy2out_(\d{8}\d{6})_(\w+).csv"
    matching_files = [file for file in files_list if re.match(pattern, file)]
    print(f'matching_files: {matching_files}')

    for filename in matching_files:
        print(f'filename: {filename}')
        match = re.match(pattern, filename)
        
        nMonteSim = int(match.group(1))
        N = int(match.group(2))
        CV = float(match.group(3))
        method = match.group(5)

        # record the datetime at the start  
        start_time = datetime.now() 
        print('start_time:', start_time) 
        print(f"Start GPM_MC_nMonteSim_{nMonteSim}_N_{N}_CV_{CV}_rpy2out_{str(start_time).split('.')[0].replace('-','').replace(' ','').replace(':','')}_{method}")

        rpy_out_files_path = os.path.join(output_folder, filename)
        # Cal the class SimulPivotMC(), generate variables in the def __init__(self)
        run = SimulPivotMC(nMonteSim, N, CV, rpy_out_files_path)  
        # start main()
        df_record, nMonte, N1, CV1, method = run.main(method=method)  
        
        # record the datetime at the end
        end_time = datetime.now() 
        # print the datetime at the end
        print('end_time:', end_time) 
        # calculate the time taken
        time_difference = end_time - start_time
        print('time_difference:', time_difference) 

            
        # output_txt1 = f"start_time: {start_time}\nend_time: {end_time}\ntime_difference: {time_difference}\n\nnMonte = {nMonte}; N1 = {N1}; CV1 = {CV1}\n\ncoverage SD: {coverage_SD}\n\ncoverage Mean: {coverage_Mean}\n"
        
        output_dir = f"MeanSD_From5Values_nMonte_{nMonte}_N_{N1}_CV_{CV1}_rpy2out_{str(end_time).split('.')[0].replace('-','').replace(' ','').replace(':','')}"
        output_dir = os.path.join(output_folder, output_dir)

        # save the results to the csv
        print('csv save to ' + output_dir + f'_{method}.csv')
        df_record.to_csv(output_dir + f'_{method}.csv')

    quit()