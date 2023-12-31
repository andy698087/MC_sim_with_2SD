import pandas as pd
import numpy as np
from scipy.stats import norm, chi2
from datetime import datetime
from math import sqrt, log, exp
import dask.dataframe as dd
# from dask import delayed
from sys import argv

class SimulPivotMC(object):
    def __init__(self, nMonteSim, N, CV):
        # number of Monte Carlo Simulation, the self. is the function to make the parameter usable across this class (module)
        self.nMonte = nMonteSim #50000>100000

        nSimulForPivot = 100000-1

        # Calculate z-score for 95% confidence interval
        self.z_score = norm.ppf(1 - 0.05 / 2)
        # print('z_score:',self.z_score)

        self.seed_value = 12181988

        # Generate random number with specified seed
        np.random.seed(self.seed_value - 1)
        self.random_numbers1_1 = np.random.rand(nSimulForPivot)

        np.random.seed(self.seed_value - 2)
        self.random_numbers1_2 = np.random.rand(nSimulForPivot)

        np.random.seed(self.seed_value - 3)
        self.random_numbers2_1 = np.random.rand(nSimulForPivot)

        np.random.seed(self.seed_value - 4)
        self.random_numbers2_2 = np.random.rand(nSimulForPivot)

        # sample size
        self.N1 = N #15, 25, 50
        self.N2 = self.N1

        # coefficient of variation
        self.CV1 = CV #0.15, 0.3, 0.5 #CV = sqrt(-1 + exp(sigma ** 2))
        self.CV2 = self.CV1

        # Mean in log scale
        self.rMeanLogScale1 = 1  #mu
        self.rMeanLogScale2 = 1
        
        # Standard deviation in log scale
        self.rSDLogScale1 = sqrt(log(1 + self.CV1 ** 2))  # sigma = sqrt(ln(1 + CV ** 2))
        # print('rSDLogScale1:', self.rSDLogScale1) # 0.47238072707743883 when CV = 0.5
        self.rSDLogScale2 = self.rSDLogScale1

    def GPM(self, row, col_SampleMeanLog1, col_SampleSDLog1, col_SampleMeanLog2, col_SampleSDLog2):
        # Generalized pivotal method, design for use for every row in a large table cosists of list of sample mean1, SD1, mean2, SD2 generated from the list of random seeds, see later in main()
        
        #group 1 pivot calculation
        SampleMeanLog1 = row[col_SampleMeanLog1]
        SampleSDLog1 = row[col_SampleSDLog1]
        # print('SampleMeanLog1:', SampleMeanLog1)
        # print('SampleSDLog1:', SampleSDLog1)
        
        U1 = np.sqrt(chi2.ppf(self.random_numbers1_1, self.N1 - 1 ))
        Z1 = norm.ppf(self.random_numbers2_1)
        # print('U1:', U1)
        # pd.DataFrame(U1).to_csv('U1_view.csv')

        Uratio1 = SampleSDLog1 * sqrt(self.N1-1)
        Zratio1 = 1/sqrt(self.N1)
        # print('Uratio1:', Uratio1)

        Pivot1 = np.exp(SampleMeanLog1- Uratio1 * Zratio1 * Z1/U1) * np.sqrt( -1 + np.exp((Uratio1/U1)**2)) * np.sqrt(np.exp((Uratio1/U1)**2))

        # print('Pivot1:', Pivot1)
        # pd.DataFrame({'SampleMeanLog1':SampleMeanLog1, 'Pivot1':Pivot1, 'part1':np.exp(SampleMeanLog1- Uratio1 * Zratio1 * Z1/U1), 'part2':np.sqrt( -1 + np.exp((Uratio1/U1)**2)),
        #               'part2-2': np.exp((Uratio1/U1)**2), 'Uratio1':Uratio1, 'U1':U1,
        #               'part3': np.sqrt(np.exp((Uratio1/U1)**2))
        #               }).to_csv('Pivot1_view.csv')
        
        #group 2 pivot calculatgition
        SampleMeanLog2 = row[col_SampleMeanLog2]
        SampleSDLog2 = row[col_SampleSDLog2]
        
        U2 = np.sqrt(chi2.ppf(self.random_numbers1_2, self.N2 - 1 ))
        Z2 = norm.ppf(self.random_numbers2_2)
        
        Uratio2 = SampleSDLog2 * sqrt(self.N2-1)
        Zratio2 = 1/sqrt(self.N2)

        Pivot2 = np.exp(SampleMeanLog2 - Uratio2 * Zratio2 * Z2/U2)  * np.sqrt( -1 + np.exp((Uratio2/U2)**2)) * np.sqrt(np.exp((Uratio2/U2)**2))
        # print("Pivot2:", Pivot2)

        result = np.log(Pivot1) - np.log(Pivot2)
        # print('result:', result)
        # pd.DataFrame(result).to_csv('result_view.csv')

        # Calculate ln ratio and SE ln ratio by percentile and Z statistics
        ln_ratio = pd.Series(result).quantile(.5)
        # print('ln_ratio:',ln_ratio)
        se_ln_ratio = (pd.Series(result).quantile(.75) - pd.Series(result).quantile(.25))/( 2 * norm.ppf(0.75))
        # print('se_ln_ratio:',se_ln_ratio)
        # Calculate the confidence intervals with z_score
        lower_bound = ln_ratio - self.z_score * se_ln_ratio
        upper_bound = ln_ratio + self.z_score * se_ln_ratio   
        # print('lower and upper bound with z_score, ln ratio and SE:',lower_bound, upper_bound)
        
        percentile_2_5 = pd.Series(result).quantile(.025)
        percentile_97_5 = pd.Series(result).quantile(.975)

        # return the lower and upper bound for determine the confidence intervals
        return lower_bound, upper_bound, ln_ratio, se_ln_ratio, percentile_2_5, percentile_97_5, 

    def Sample_inv_normal(self, row, seed_):
        # calculate the mean and std of a sample generate from random generater of normal distribution
        np.random.seed(row[seed_]) # use seed from pre-determined list
        rSampleOfRandoms = [exp(norm.ppf(i,loc=self.rMeanLogScale1, scale=self.rSDLogScale1)) for i in np.random.rand(self.N1+self.N2)] # generate log-normal distribution
        # print('rSampleOfRandoms:',rSampleOfRandoms)
    
        return rSampleOfRandoms
    
    def Sample_inv_normal_no_exp(self, row, seed_):
        # calculate the mean and std of a sample generate from random generater of normal distribution
        np.random.seed(row[seed_]) # use seed from pre-determined list
        rSampleOfRandoms = [(norm.ppf(i,loc=self.rMeanLogScale1, scale=self.rSDLogScale1)) for i in np.random.rand(self.N1+self.N2)] # generate log-normal distribution
        # print('rSampleOfRandoms:',rSampleOfRandoms)
    
        return rSampleOfRandoms

    def Sample_uniformity(self, row, seed_):
        # calculate the mean and std of a sample generate from random generater of normal distribution
        np.random.seed(row[seed_]) # use seed from pre-determined list
        rSampleOfRandoms = [i for i in np.random.rand(self.N1+self.N2)] # generate log-normal distribution
        # print('rSampleOfRandoms:',rSampleOfRandoms)
    
        return rSampleOfRandoms

    def Mean_SD(self, row):
        half_N1_N2 = int(0.5 * (self.N1+self.N2))
        
        rSampleOfRandoms1 = row[:half_N1_N2]
        rSampleOfRandoms2 = row[half_N1_N2:]

        rSampleMean1 = np.mean(rSampleOfRandoms1)
        rSampleSD1 = np.std(rSampleOfRandoms1, ddof=1)

        rSampleMeanLogScale1 = log(rSampleMean1) - 0.5 * log(1 + (rSampleSD1/rSampleMean1)**2)
        rSampleSDLogScale1 = sqrt(log(1 + (rSampleSD1/rSampleMean1)**2))

        rSampleMean2 = np.mean(rSampleOfRandoms2)
        rSampleSD2 = np.std(rSampleOfRandoms2, ddof=1)

        rSampleMeanLogScale2 = log(rSampleMean2) - 0.5 * log(1 + (rSampleSD2/rSampleMean2)**2)
        rSampleSDLogScale2 = sqrt(log(1 + (rSampleSD2/rSampleMean2)**2))

        return rSampleMeanLogScale1, rSampleSDLogScale1, rSampleMeanLogScale2, rSampleSDLogScale2 #, rSampleMean1, rSampleSD1, rSampleMean2, rSampleSD2

    def Mean_SD_for_LogScale(self, row): # noMethodOfMoment
        half_N1_N2 = int(0.5 * (self.N1+self.N2))
        
        rSampleOfRandoms1 = row[:half_N1_N2]
        rSampleOfRandoms2 = row[half_N1_N2:]
        
        rSampleMeanLogScale1 = np.mean(rSampleOfRandoms1)
        rSampleSDLogScale1 = np.std(rSampleOfRandoms1, ddof=1)
        rSampleMeanLogScale2 = np.mean(rSampleOfRandoms2)
        rSampleSDLogScale2 = np.std(rSampleOfRandoms2, ddof=1)

        return rSampleMeanLogScale1, rSampleSDLogScale1, rSampleMeanLogScale2, rSampleSDLogScale2 #, rSampleMean1, rSampleSD1, rSampleMean2, rSampleSD2

    def Mean_SD_higgins1(self, row):
        half_N1_N2 = int(0.5 * (self.N1+self.N2))
        
        rSampleOfRandoms1 = row[:half_N1_N2]
        rSampleOfRandoms2 = row[half_N1_N2:]

        rSampleMean1 = np.mean(rSampleOfRandoms1)
        rSampleSD1 = np.std(rSampleOfRandoms1, ddof=1)

        # Given values
        # Substitute A5, B5, and C5 with self.N1, rSampleMean1, and rSampleSD1

        rSampleMeanLogScale1 = log(rSampleMean1) - 0.5 * log(1 + (rSampleSD1/rSampleMean1)**2)
        rSampleSDLogScale1, _ = self.taylor(self.N1, rSampleMean1, rSampleSD1)

        rSampleMean2 = np.mean(rSampleOfRandoms2)
        rSampleSD2 = np.std(rSampleOfRandoms2, ddof=1)

        rSampleMeanLogScale2 = log(rSampleMean2) - 0.5 * log(1 + (rSampleSD2/rSampleMean2)**2)
        rSampleSDLogScale2, _ = self.taylor(self.N2, rSampleMean2, rSampleSD2)
        
        return rSampleMeanLogScale1, rSampleSDLogScale1, rSampleMeanLogScale2, rSampleSDLogScale2 #, rSampleMean1, rSampleSD1, rSampleMean2, rSampleSD2
    
    def Mean_SD_higgins2(self, row):
        half_N1_N2 = int(0.5 * (self.N1+self.N2))
        
        rSampleOfRandoms1 = row[:half_N1_N2]
        rSampleOfRandoms2 = row[half_N1_N2:]

        rSampleMean1 = np.mean(rSampleOfRandoms1)
        rSampleSD1 = np.std(rSampleOfRandoms1, ddof=1)

        # Given values
        # Substitute A5, B5, and C5 with self.N1, rSampleMean1, and rSampleSD1

        rSampleSDLogScale1, rSampleMeanLogScale1 = self.taylor(self.N1, rSampleMean1, rSampleSD1)

        rSampleMean2 = np.mean(rSampleOfRandoms2)
        rSampleSD2 = np.std(rSampleOfRandoms2, ddof=1)

        rSampleSDLogScale2, rSampleMeanLogScale2 = self.taylor(self.N2, rSampleMean2, rSampleSD2)

        return rSampleMeanLogScale1, rSampleSDLogScale1, rSampleMeanLogScale2, rSampleSDLogScale2 #, rSampleMean1, rSampleSD1, rSampleMean2, rSampleSD2
    
    def taylor(self, N, Mean, SD):
        MeanLogScale_1 = log(Mean) - (1/2) * log(1 + (SD/N)**2)   #Mean
        SDLogScale_1 = sqrt(log((SD/Mean)**2 + 1))   #SD
        var_x = (SD**2) / N
        dz_dmean = (1/Mean) + (SD**2) / (Mean * ((SD**2) + (Mean**2)))
        cov_x_sx2 = (1/N) * exp(3*MeanLogScale_1) * (exp(9*(SDLogScale_1**2)/2) - 3*exp(5*(SDLogScale_1**2)/2) + 2*exp(3*(SDLogScale_1**2)/2))
        dz_dsx2 = -1 / (2 * ((SD**2) + (Mean**2)))
        var_sx2 = (1/N) * exp(4*MeanLogScale_1) * (exp(8*SDLogScale_1**2) - 4*exp(5*SDLogScale_1**2) - exp(4*SDLogScale_1**2) + 8*exp(3*SDLogScale_1**2) - 4*exp(2*SDLogScale_1**2))
        var_B_z = var_x * dz_dmean**2 + 2 * cov_x_sx2 * dz_dmean * dz_dsx2 + var_sx2 * dz_dsx2**2
        SDLogScale_2 = sqrt(N * var_B_z)   #SD
        MeanLogScale_2 = abs(Mean - exp(MeanLogScale_1 + (SDLogScale_2**2)/2)) / Mean   #Mean
        return SDLogScale_2, MeanLogScale_2

    def Coverage(self, row, lower_bound, upper_bound): #nMonte = 50000, 
        # Check how many confidence intervals include 0
        lower_bound = row[lower_bound]
        upper_bound = row[upper_bound]

        intervals_include_zero = (lower_bound <= 0) and (upper_bound >= 0)
        return int(intervals_include_zero)  # 1 as True, 0 as False, check_tolerance
       
    def main(self, method_of_moments = True, expanding_rSampleOfRandoms = False): # the main process, method of moments = False, True, 'Higgins1', 'Higgins2'
        list_seeds = [i for i in range(self.seed_value, self.seed_value + self.nMonte)] # the pre-determined list of seeds, number of nMonte
        
        df = pd.DataFrame({'Seeds':list_seeds}) # put the list of seeds into a table (a.k.a DataFrame) with one column named "Seeds"
        # df_record[['N1', 'N2']] = [[self.N1, self.N2] for _ in range(self.nMonte)]       
        if method_of_moments:
            df['rSampleOfRandoms']  = df.apply(self.Sample_inv_normal, args=('Seeds',), axis=1) # generate random sample inv normal(above)
            df_record = df
            df = dd.from_pandas(df['rSampleOfRandoms'], npartitions=35) # put the table into dask, a progress that can parallel calculating each rows with multi-thread
            if method_of_moments == 'Higgins1':
                print('Higgins1')
                df = df.apply(self.Mean_SD_higgins1, meta=('float64', 'float64')) # generate sample mean and SD in Log scale using Mean_SD_higgins1 (above)
            elif method_of_moments == 'Higgins2':
                print('Higgins2')
                df = df.apply(self.Mean_SD_higgins2, meta=('float64', 'float64')) # generate sample mean and SD in Log scale using Mean_SD_higgins2 (above)
            else:
                print('original MoM')
                df = df.apply(self.Mean_SD, meta=('float64', 'float64')) # generate sample mean and SD in Log scale using Mean_SD (above)
        else:
            print('no MoM')
            df['rSampleOfRandoms'] = df.apply(self.Sample_inv_normal_no_exp, args=('Seeds',), axis=1) # generate random sample inv normal(above)   
            df_record = df
            df = dd.from_pandas(df['rSampleOfRandoms'], npartitions=50) # put the table into dask, a progress that can parallel calculating each rows with multi-thread
            df = df.apply(self.Mean_SD_for_LogScale, meta=('float64', 'float64')) # generate sample mean and SD in Log scale using Sample_Mean_SD (above)
       
        df_record[['rSampleMeanLogScale1', 'rSampleSDLogScale1', 'rSampleMeanLogScale2', 'rSampleSDLogScale2']] = df.compute().tolist()
        
        df = df.apply(self.GPM, args=(0,1,2,3), meta=('float64', 'float64'))  # generate lower and upper bounds with sample mean and SD using GPM (above) # the slowest part
        df_record[['lower_bound_SE', 'upper_bound_SE', 'ln_ratio', 'se_ln_ratio', 'percentile_2_5', 'percentile_97_5']] = df.compute().tolist()

        df = df.apply(self.Coverage, args=(0,1), meta = ('float64', 'float64')) # check coverage of each rows
        df_record['intervals_include_zero'] = df.compute().tolist()

        coverage_by_ln_ratio = df.mean().compute() # compute the mean of the list of coverage (0 or 1), it equal to the percentage of coverage

        if expanding_rSampleOfRandoms:
            print('expanding_rSampleOfRandoms')
            # Use apply to expand the 'rSampleOfRandoms' column into separate columns
            expanded_df = df_record['rSampleOfRandoms'].apply(pd.Series)

            # Rename the columns if needed
            expanded_df.columns = [f'rSampleOfRandoms_{i}' for i in range(1, self.N1 + self.N2 + 1)]

            # Reverse the insertion sequence by reversing the order of columns in expanded_df
            expanded_df = expanded_df[expanded_df.columns[::-1]]

            # Get the original position of the columns you plan to drop
            original_positions = df_record.columns.get_loc('rSampleOfRandoms')
            original_positions = [original_positions for i in range(len(expanded_df.columns))]

            # df_record = df_record.join(expanded_df)
            df_record.drop(columns = 'rSampleOfRandoms', inplace=True)        

            df_record = pd.concat([df_record, expanded_df], axis=1)

        return coverage_by_ln_ratio, df_record, self.nMonte, self.N1, self.CV1
    

    def main_uniformity(self):
        list_seeds = [i for i in range(self.seed_value, self.seed_value + self.nMonte)] # the pre-determined list of seeds, number of nMonte
        
        df = pd.DataFrame({'Seeds':list_seeds}) # put the list of seeds into a table (a.k.a DataFrame) with one column named "Seeds"
        # df_record[['N1', 'N2']] = [[self.N1, self.N2] for _ in range(self.nMonte)]
        df['rSampleOfRandoms'] = df.apply(self.Sample_uniformity, args=('Seeds',), axis=1) # generate random sample inv normal(above)
        return df['rSampleOfRandoms']
