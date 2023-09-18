import pandas as pd
import numpy as np
from scipy.stats import norm, chi2
from datetime import datetime
from math import sqrt, log, exp
import dask.dataframe as dd

class SimulPivotMC(object):
    def __init__(self, nMonteSim, N, CV):
        # number of Monte Carlo Simulation
        self.nMonte = nMonteSim #50000>100000

        # Calculate z-score for alpha = 0.05, 
        # ppf is the percent point function that is inverse of cumulative distribution function
        self.z_score = norm.ppf(1 - 0.05 / 2)

        # Sample size, we choose 15, 25, 50, notation "n" in the manuscript
        self.N1 = N #15, 25, 50
        self.N2 = self.N1

        # coefficient of variation, we choose 0.15, 0.3, 0.5
        self.CV1 = CV
        self.CV2 = self.CV1

        # Mean in log scale, notation "μ_i" in the manuscript
        self.rMeanLogScale1 = 1  #mu
        self.rMeanLogScale2 = self.rMeanLogScale1
        
        # Standard deviation in log scale, notation "σ_i" in the manuscript
        self.rSDLogScale1 = sqrt(log(1 + self.CV1 ** 2)) #Equation 1 in the manuscript
        self.rSDLogScale2 = self.rSDLogScale1


        # the number for pivot, the notation "m" in the manuscript
        nSimulForPivot = 100000-1

        # choosing a seed
        self.seed_value = 12181988

        # Generate 4 set of random numbers each with specified seed
        np.random.seed(self.seed_value - 1)
        random_numbers1_1 = np.random.rand(nSimulForPivot)

        np.random.seed(self.seed_value - 2)
        random_numbers1_2 = np.random.rand(nSimulForPivot)

        np.random.seed(self.seed_value - 3)
        random_numbers2_1 = np.random.rand(nSimulForPivot)

        np.random.seed(self.seed_value - 4)
        random_numbers2_2 = np.random.rand(nSimulForPivot)

        # Generate random number for later used in calculating Ui and Zi in generalized pivotal method
        # group 1 pivot calculation
        # U_i and Z_i used in Equation 3
        self.U1 = chi2.ppf(random_numbers1_1, self.N1 - 1 )
        self.Z1 = norm.ppf(random_numbers2_1)

        #group 2 pivot calculation
        # U_i and Z_i used in Equation 3
        self.U2 = chi2.ppf(random_numbers1_2, self.N2 - 1 )
        self.Z2 = norm.ppf(random_numbers2_2)
       
    def main(self, method_of_moments): # the main process, method of moments = ['no_moments', 'first_two_moment', 'higher_orders_of_moments']
        list_seeds = [i for i in range(self.seed_value, self.seed_value + self.nMonte)] # the pre-determined list of seeds, using number of nMonte
        df = pd.DataFrame({'Seeds':list_seeds}) # put the list of seeds into a table (a.k.a DataFrame) with one column named "Seeds"  

        if method_of_moments == 'no_moments': #using no method of moments 
            # generate log-normal distribution, using mean of rMeanLogScale and standard deviation of rSDLogScale
            df['rSampleOfRandoms'] = df.apply(self.Sample_inv_normal, args=('Seeds',), axis=1) 

            df_record = df
            df = dd.from_pandas(df['rSampleOfRandoms'], npartitions=35) # put the table into dask, a progress that can parallel calculating each rows with multi-thread
            
            df = df.apply(self.Mean_SD, meta=('float64', 'float64')) # generate sample mean and SD in Log scale using Sample_Mean_SD (above)

        elif method_of_moments in ['first_two_moment', 'higher_orders_of_moments', 'higher_orders_of_moments_mean']:
            # generate log-normal distribution, using mean of rMeanLogScale and standard deviation of rSDLogScale
            df['rSampleOfRandoms'] = df.apply(self.Sample_inv_normal, args=('Seeds',), axis=1).apply(lambda x: [np.exp(item) for item in x])
            df_record = df
            df = dd.from_pandas(df['rSampleOfRandoms'], npartitions=35) # put the table into dask, a progress that can parallel calculating each rows with multi-thread
            
            df = df.apply(self.Mean_SD, meta=('float64', 'float64')) # generate sample mean and SD in Log scale using Sample_Mean_SD (above)

            if method_of_moments == 'first_two_moment':
                
                df = df.apply(self.first_two_moment, args=(0,1,2,3), meta=('float64', 'float64')) # generate sample mean and SD in Log scale using Mean_SD (above)
            
            elif method_of_moments == 'higher_orders_of_moments':
                
                df = df.apply(self.higher_orders_of_moments, args=(0,1,2,3), meta=('float64', 'float64')) # generate sample mean and SD in Log scale using Mean_SD_higgins1 (above)
            
            elif method_of_moments == 'higher_orders_of_moments_mean':
                
                df = df.apply(self.higher_orders_of_moments_mean, args=(0,1,2,3), meta=('float64', 'float64')) # generate sample mean and SD in Log scale using Mean_SD_higgins1 (above)
       
        df_record[['rSampleMeanLogScale1', 'rSampleSDLogScale1', 'rSampleMeanLogScale2', 'rSampleSDLogScale2']] = df.compute().tolist()
        
        # Equation 3
        df = df.apply(self.GPM_log_ratio_SD, args=(0,1,2,3), meta=('float64', 'float64'))  # generate lower and upper bounds with sample mean and SD using GPM (above) # the slowest part
        df_record[['ln_ratio', 'se_ln_ratio']] = df.compute().tolist()

        df = df.apply(self.Coverage, args=(0,1), meta = ('float64', 'float64')) # check coverage of each rows
        df_record['intervals_include_zero'] = df.compute().tolist()

        coverage = df.mean().compute() # compute the mean of the list of coverage (0 or 1), it equal to the percentage of coverage

        return coverage, df_record, self.nMonte, self.N1, self.CV1, method_of_moments

    
    def Sample_inv_normal(self, row, seed_):
        # generate log-normal distribution, using mean of rMeanLogScale and standard deviation of rSDLogScale
        np.random.seed(row[seed_]) # using seed from pre-determined list
        rSampleOfRandoms = [(norm.ppf(i,loc=self.rMeanLogScale1, scale=self.rSDLogScale1)) for i in np.random.rand(self.N1+self.N2)] # generate log-normal distribution
        
        return rSampleOfRandoms

    def Mean_SD(self, row): # noMethodOfMoment
        half_N1_N2 = int(0.5 * (self.N1+self.N2))
        
        rSampleOfRandoms1 = row[:half_N1_N2]
        rSampleOfRandoms2 = row[half_N1_N2:]
        
        rSampleMean1 = np.mean(rSampleOfRandoms1)
        rSampleSD1 = np.std(rSampleOfRandoms1, ddof=1)
        rSampleMean2 = np.mean(rSampleOfRandoms2)
        rSampleSD2 = np.std(rSampleOfRandoms2, ddof=1)

        return rSampleMean1, rSampleSD1, rSampleMean2, rSampleSD2 #, rSampleMean1, rSampleSD1, rSampleMean2, rSampleSD2

    def first_two_moment(self, row, col_SampleMean1, col_SampleSD1, col_SampleMean2, col_SampleSD2): #using Equation 7 and 8
        
        SampleMean1 = row[col_SampleMean1]
        SampleSD1 = row[col_SampleSD1]

        SampleMean2 = row[col_SampleMean2]
        SampleSD2 = row[col_SampleSD2]

        rSampleMeanLogScale1, rSampleSDLogScale1, _, _ = self.transform_from_raw_to_log_mean_SD(self.N1, SampleMean1, SampleSD1)
        rSampleMeanLogScale2, rSampleSDLogScale2, _, _ = self.transform_from_raw_to_log_mean_SD(self.N2, SampleMean2, SampleSD2)
                   
        return rSampleMeanLogScale1, rSampleSDLogScale1, rSampleMeanLogScale2, rSampleSDLogScale2 #, rSampleMean1, rSampleSD1, rSampleMean2, rSampleSD2

    def higher_orders_of_moments(self, row, col_SampleMean1, col_SampleSD1, col_SampleMean2, col_SampleSD2):#using Equation 7 and 9
        
        SampleMean1 = row[col_SampleMean1]
        SampleSD1 = row[col_SampleSD1]

        SampleMean2 = row[col_SampleMean2]
        SampleSD2 = row[col_SampleSD2]
        
        rSampleMeanLogScale1, _, rSampleSDLogScale1, _ = self.transform_from_raw_to_log_mean_SD(self.N1, SampleMean1, SampleSD1)
        rSampleMeanLogScale2, _, rSampleSDLogScale2, _ = self.transform_from_raw_to_log_mean_SD(self.N2, SampleMean2, SampleSD2)

        return rSampleMeanLogScale1, rSampleSDLogScale1, rSampleMeanLogScale2, rSampleSDLogScale2 #, rSampleMean1, rSampleSD1, rSampleMean2, rSampleSD2

    def higher_orders_of_moments_mean(self, row, col_SampleMean1, col_SampleSD1, col_SampleMean2, col_SampleSD2):#using Equation 7 and 9
        
        SampleMean1 = row[col_SampleMean1]
        SampleSD1 = row[col_SampleSD1]

        SampleMean2 = row[col_SampleMean2]
        SampleSD2 = row[col_SampleSD2]
        
        _, _, rSampleSDLogScale1, rSampleMeanLogScale1 = self.transform_from_raw_to_log_mean_SD(self.N1, SampleMean1, SampleSD1)
        _, _, rSampleSDLogScale2, rSampleMeanLogScale2 = self.transform_from_raw_to_log_mean_SD(self.N2, SampleMean2, SampleSD2)

        return rSampleMeanLogScale1, rSampleSDLogScale1, rSampleMeanLogScale2, rSampleSDLogScale2 #, rSampleMean1, rSampleSD1, rSampleMean2, rSampleSD2


    #Equation 7, 8, and 9
    def transform_from_raw_to_log_mean_SD(self, N, Mean, SD):
        MeanLogScale_1 = log(Mean) - (1/2) * log(1 + (SD/Mean)**2)   #Mean in log scale, Equation 7
        SDLogScale_1 = sqrt(log((SD/Mean)**2 + 1))   #SD in log scale, Equation 8
        var_x = (SD**2) / N
        dz_dmean = (1/Mean) + (SD**2) / (Mean * ((SD**2) + (Mean**2)))
        cov_x_sx2 = (1/N) * exp(3*MeanLogScale_1) * (exp(9*(SDLogScale_1**2)/2) - 3*exp(5*(SDLogScale_1**2)/2) + 2*exp(3*(SDLogScale_1**2)/2))
        dz_dsx2 = -1 / (2 * ((SD**2) + (Mean**2)))
        var_sx2 = (1/N) * exp(4*MeanLogScale_1) * (exp(8*SDLogScale_1**2) - 4*exp(5*SDLogScale_1**2) - exp(4*SDLogScale_1**2) + 8*exp(3*SDLogScale_1**2) - 4*exp(2*SDLogScale_1**2))
        var_B_z = var_x * dz_dmean**2 + 2 * cov_x_sx2 * dz_dmean * dz_dsx2 + var_sx2 * dz_dsx2**2
        SDLogScale_2 = sqrt(N * var_B_z)   #SD in log scale, Equation 9
        MeanLogScale_2 = abs(Mean - exp(MeanLogScale_1 + (SDLogScale_2**2)/2)) / Mean   #Mean
        return MeanLogScale_1, SDLogScale_1, SDLogScale_2, MeanLogScale_2
    
    def GPM_log_ratio_SD(self, row, col_SampleMeanLog1, col_SampleSDLog1, col_SampleMeanLog2, col_SampleSDLog2):  # Equation 2 and 3 

        #group 1 pivot calculation
        SampleMeanLog1 = row[col_SampleMeanLog1]
        SampleSDLog1 = row[col_SampleSDLog1]
        Pivot1 = self.Pivot_calculation(SampleMeanLog1, SampleSDLog1, self.N1, self.U1, self.Z1)
        #group 2 pivot calculation
        SampleMeanLog2 = row[col_SampleMeanLog2]
        SampleSDLog2 = row[col_SampleSDLog2]
        Pivot2 = self.Pivot_calculation(SampleMeanLog2, SampleSDLog2, self.N2, self.U2, self.Z2)

        # Equation 2, generalized pivotal statistics
        pivot_statistics = np.log(Pivot1) - np.log(Pivot2)

        # Calculate ln ratio and SE ln ratio by percentile and Z statistics, Equation 4 and 5
        ln_ratio = pd.Series(pivot_statistics).quantile(.5)
        se_ln_ratio = (pd.Series(pivot_statistics).quantile(.75) - pd.Series(pivot_statistics).quantile(.25))/(norm.ppf(.75) - norm.ppf(.25))
        
        return ln_ratio, se_ln_ratio
    
    # Equation 3
    def Pivot_calculation(self, rSampleMeanLogScale, rSampleSDLogScale, N, U, Z):
        return np.exp(rSampleMeanLogScale- np.sqrt((rSampleSDLogScale**2 * (N-1))/U) * (Z/sqrt(N)) ) *\
                    np.sqrt((np.exp((rSampleSDLogScale**2 * (N-1))/U) - 1) * np.exp((rSampleSDLogScale**2 * (N-1))/U))

    def Coverage(self, row, col_ln_ratio, col_se_ln_ratio): #nMonte = 50000, 
        
        ln_ratio = row[col_ln_ratio]
        se_ln_ratio = row[col_se_ln_ratio]

        # Calculate the confidence intervals with z_score of alpha = 0.05, Equation 6
        lower_bound = ln_ratio - self.z_score * se_ln_ratio
        upper_bound = ln_ratio + self.z_score * se_ln_ratio   

        intervals_include_zero = (lower_bound <= 0) and (upper_bound >= 0)
        return int(intervals_include_zero)  # 1 as True, 0 as False, check_tolerance
    
if __name__ == '__main__':
    nMonteSim = 100000 # nMonte
    for method_of_moments in ['first_two_moment','no_moments']:#,'higher_orders_of_moments', 'higher_orders_of_moments_mean']:
        print(method_of_moments)
        for N in [15, 25, 50]:
            for CV in [0.15, 0.3, 0.5]:
                start_time = datetime.now() # record the datetime at the start
                print('start_time:', start_time) # print the datetime at the start
                print(f"Start GPM_MC_nMonteSim_{nMonteSim}_N_{N}_CV_{CV}_{str(start_time).split('.')[0].replace('-','').replace(' ','').replace(':','')}_{method_of_moments}")

                run = SimulPivotMC(nMonteSim, N, CV)  # Generate the class SimulPivotMC(), generate variables in the def __init__(self)
                coverage_by_ln_ratio, df_record, nMonte, N1, CV1, method_of_moments = run.main(method_of_moments=method_of_moments)  # start main()
                end_time = datetime.now() # record the datetime at the end
                print('end_time:', end_time) # print the datetime at the end
                time_difference = end_time - start_time
                print('time_difference:', time_difference) # calculate the time taken

                print('percentage coverage: %s' %(coverage_by_ln_ratio,)) # print out the percentage of coverage
                    
                output_txt1 = f"start_time: {start_time}\nend_time: {end_time}\ntime_difference: {time_difference}\n\nnMonte = {nMonte}; N1 = {N1}; CV= {CV1}\n\nln ratio SE include zero: {coverage_by_ln_ratio}\n"
                
                output_dir = f"GPM_MC_nMonte_{nMonte}_N_{N1}_CV_{CV1}_{str(end_time).split('.')[0].replace('-','').replace(' ','').replace(':','')}"
                
                print('csv save to ' + output_dir + f'_{method_of_moments}.csv')
                df_record.to_csv(output_dir + f'_{method_of_moments}.csv')

                with open(output_dir + f'_{method_of_moments}.txt', 'w') as f:
                    f.write(output_txt1)
    quit()