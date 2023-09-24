#This one, using the multithread, runs quickly (about 10 min for each 100,000 simulations). It obtains results identical to at least 16 digits as the one using no multithread.
import pandas as pd
import numpy as np
from scipy.stats import norm, chi2, weibull_min
from datetime import datetime
from math import sqrt, log, exp, gamma
import dask.dataframe as dd
from scipy.optimize import minimize

class SimulPivotMC(object):
    def __init__(self, nMonteSim, N, CVTimeScale):
        # number of Monte Carlo Simulation
        self.nMonte = nMonteSim

        # Calculate z-score for alpha = 0.05, ppf is the percent point function that is inverse of cumulative distribution function
        self.z_score = norm.ppf(1 - 0.05 / 2)

        # Sample size, we choose 15, 25, 50, notation "n" in the manuscript
        self.N1 = N
        self.N2 = self.N1

        # coefficient of variation, we choose 0.15, 0.3, 0.5
        self.CV1 = CVTimeScale
        self.CV2 = self.CV1

        # Mean in log scale, notation "μ_i" in the manuscript
        MeanTimeScale = 1
        self.rMeanLogScale1 = log(MeanTimeScale)
        # print('self.rMeanLogScale1:',self.rMeanLogScale1 )
        self.rMeanLogScale2 = self.rMeanLogScale1
        
        # Standard deviation in log scale, notation "σ_i" in the manuscript, Equation 1 in the manuscript
        self.rSDLogScale1 = sqrt(log(1 + self.CV1 ** 2)) 
        self.rSDLogScale2 = self.rSDLogScale1
        # print('self.rSDLogScale1:',self.rSDLogScale1)

        df_weibull_params = pd.read_csv('weibull_params1e-6.csv')[['CV','shape_parameter','scale_parameter']]
        self.shape_parameter, self.scale_parameter = df_weibull_params[df_weibull_params['CV']==CVTimeScale][['shape_parameter','scale_parameter']].iloc[0,:]

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

    def main(self):
        # the pre-determined list of seeds, using number of nMonte
        list_seeds = [i for i in range(self.seed_value, self.seed_value + self.nMonte)] 
        # put the list of seeds into a table (a.k.a DataFrame) with one column named "Seeds"  
        print('Sample_Weibull')
        df = pd.DataFrame({'Seeds':list_seeds}) 
        df['rSampleOfRandomsWeibull'] = df.apply(self.Sample_Weibull, args=('Seeds',), axis=1)
        df_record = df
        # df = df['rSampleOfRandomsLogNorm'].copy()
        # put the table into dask, a progress that can parallel calculating each rows using multi-thread
        df = dd.from_pandas(df['rSampleOfRandomsWeibull'], npartitions=35) 
        meta = ('float64', 'float64')
        print('Mean_SD')
        # calculate sample mean and Var using Mean_SD
        df = df.apply(self.Mean_SD, meta=meta)
        df_record[['WeibullMean1', 'WeibullSD1', 'WeibullMean2', 'WeibullSD2']] = df.compute().tolist()
        # df_record[['LogNormMean1', 'LogNormVar1', 'LogNormMean2', 'LogNormVar2']] = df.apply(lambda x: pd.Series(x))
        # print(df_record)
        # print(df)
        
        print('first_two_moment')
        # using first_two_moment to transform from raw to log mean and SD                        
        # using Equation 7 and 8
        # generate sample mean and SD in Log scale using Mean_SD
        df = df.apply(self.first_two_moment, args=(0,1,2,3), meta=meta) 
        df_record[['rSampleMeanLogScale1', 'rSampleSDLogScale1', 'rSampleMeanLogScale2', 'rSampleSDLogScale2']] = df.compute().tolist()
        
        print('GPM_log_ratio_SD')
        # Equation 3 # generate 'ln_ratio' and 'se_ln_ratio' with sample mean and SD using GPM
        df = df.apply(self.GPM_log_ratio_SD, args=(0,1,2,3), meta=meta)  
        df_record[['ln_ratio', 'se_ln_ratio']] = df.compute().tolist()
        
        print('Coverage')
        # check coverage of each rows
        df = df.apply(self.Coverage, args=(0,1), meta=meta) 
        df_record['intervals_include_zero'] = df.compute().tolist()
        print('compute dask')
        # compute the mean of the list of coverage (0 or 1), it equals to the percentage of coverage
        coverage = df.mean().compute() 

        return coverage, df_record, self.nMonte, self.N1, self.CV1


    def Sample_Weibull(self, row, seed_):
        rSampleOfRandoms = weibull_min.rvs(self.shape_parameter, scale=self.scale_parameter, size=self.N1+self.N2, random_state = row[seed_])
        # np.random.seed(row[seed_])
        # rSampleOfRandoms = [(weibull_min.ppf(i, self.shape_parameter, scale=self.scale_parameter)) for i in np.random.rand(self.N1+self.N2)]

        return rSampleOfRandoms

    def Sample_inv_normal(self, row, seed_):
        # generate log-normal distribution, using mean of rMeanLogScale and standard deviation of rSDLogScale
        # rSampleOfRandoms = norm.rvs(loc=self.rMeanLogScale1, scale=self.rSDLogScale1, size=self.N1+self.N2, random_state = row[seed_])
        np.random.seed(row[seed_])
        rSampleOfRandoms = np.exp([(norm.ppf(i,loc=self.rMeanLogScale1, scale=self.rSDLogScale1)) for i in np.random.rand(self.N1+self.N2)] )

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

    def Mean_Var(self, row):
        # print('N1:', N1)
        rSampleOfRandoms1 = row[:self.N1]
        # print('rSampleOfRandoms1:',rSampleOfRandoms1)
        # the mean of rSampleOfRandoms1, notation "z_i"
        rSampleMean1 = np.mean(rSampleOfRandoms1)  
        # the standard deviation of rSampleOfRandoms1, delta degree of freeden = 1, notation "sz_i"
        rSampleVar1 = np.var(rSampleOfRandoms1, ddof=1) 
        
        if N2 != 0:
            # print('N2:', N2)
            rSampleOfRandoms2 = row[self.N1:(self.N1+self.N2)]
            # print('rSampleOfRandoms2:',rSampleOfRandoms2)
            rSampleMean2 = np.mean(rSampleOfRandoms2)
            rSampleVar2 = np.var(rSampleOfRandoms2, ddof=1)
            return rSampleMean1, rSampleVar1, rSampleMean2, rSampleVar2
        else:
            return rSampleMean1, rSampleVar1
              
    def first_two_moment(self, row, col_SampleMean1, col_SampleSD1, col_SampleMean2, col_SampleSD2):
        
        SampleMean1 = row[col_SampleMean1]
        SampleSD1 = row[col_SampleSD1]

        SampleMean2 = row[col_SampleMean2]
        SampleSD2 = row[col_SampleSD2]

        #using Equation 7 and 8
        rSampleMeanLogScale1, rSampleSDLogScale1, _ = self.transform_from_raw_to_log_mean_SD(SampleMean1, SampleSD1)
        rSampleMeanLogScale2, rSampleSDLogScale2, _ = self.transform_from_raw_to_log_mean_SD(SampleMean2, SampleSD2)

        return rSampleMeanLogScale1, rSampleSDLogScale1, rSampleMeanLogScale2, rSampleSDLogScale2

    def transform_from_raw_to_log_mean_SD(self, Mean, SD):
        CV = SD/Mean
        CVsq = CV**2
        #Mean in log scale, Equation 7
        MeanLogScale_1 = log(Mean/sqrt(CVsq + 1)) 
        #SD in log scale, Equation 8
        SDLogScale_1 = sqrt(log((CVsq + 1)))
        #SD in log scale, Equation 9
        SDLogScale_2 = sqrt(CVsq * (1 + CVsq/(1 + CVsq))**2 - (1 + CVsq/(1 + CVsq)) * ((1 + CVsq)**2 - 3 + 2/(1 + CVsq)) + (1/4) * ((1 + CVsq)**4 - 4*(1 + CVsq) - 1 + 8/(1 + CVsq) - 4/((1 + CVsq)**2)))   
        return MeanLogScale_1, SDLogScale_1, SDLogScale_2
    
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
     
    def Pivot_calculation(self, rSampleMeanLogScale, rSampleSDLogScale, N, U, Z):
        # Equation 3
        return np.exp(rSampleMeanLogScale- np.sqrt((rSampleSDLogScale**2 * (N-1))/U) * (Z/sqrt(N)) ) * np.sqrt((np.exp((rSampleSDLogScale**2 * (N-1))/U) - 1) * np.exp((rSampleSDLogScale**2 * (N-1))/U))

    def Coverage(self, row, col_ln_ratio, col_se_ln_ratio):
        
        ln_ratio = row[col_ln_ratio]
        se_ln_ratio = row[col_se_ln_ratio]

        # Calculate the confidence intervals with z_score of alpha = 0.05, Equation 6
        lower_bound = ln_ratio - self.z_score * se_ln_ratio
        upper_bound = ln_ratio + self.z_score * se_ln_ratio   

        intervals_include_zero = (lower_bound < 0) and (upper_bound > 0)
        # 1 as True, 0 as False, check coverage
        return int(intervals_include_zero)  

        
if __name__ == '__main__':
    # number of Monte Carlo simulations
    nMonteSim = 10
    # Sample size, we choose 15, 25, 50, notation "n" in the manuscript
    for N in [25]: 
        # coefficient of variation, we choose 0.15, 0.3, 0.5
        for CV in [0.3]: 
            # record the datetime at the start
            start_time = datetime.now() 
            print('start_time:', start_time) 
            print(f"Start GPM_MC_nMonteSim_{nMonteSim}_N_{N}_CV_{CV}_{str(start_time).split('.')[0].replace('-','').replace(' ','').replace(':','')}")

            # Cal the class SimulPivotMC(), generate variables in the def __init__(self)
            run = SimulPivotMC(nMonteSim, N, CV)  
            # start main()
            coverage_by_ln_ratio, df_record, nMonte, N1, CV1 = run.main()  
            
            # record the datetime at the end
            end_time = datetime.now() 
            # print the datetime at the end
            print('end_time:', end_time) 
            # calculate the time taken
            time_difference = end_time - start_time
            print('time_difference:', time_difference) 
            # print out the percentage of coverage
            print('percentage coverage: %s' %(coverage_by_ln_ratio,)) 
                
            output_txt1 = f"start_time: {start_time}\nend_time: {end_time}\ntime_difference: {time_difference}\n\nnMonte = {nMonte}; N1 = {N1}; CV1 = {CV1}\n\n percentage coverage: {coverage_by_ln_ratio}\n"
            
            output_dir = f"Weibull_GPM_MC_nMonte_{nMonte}_N_{N1}_CV_{CV1}_{str(end_time).split('.')[0].replace('-','').replace(' ','').replace(':','')}"
            
            # save the results to the csv
            print('csv save to ' + output_dir + f'_.csv')
            df_record.to_csv(output_dir + f'_.csv')

            # save the results to the txt
            with open(output_dir + f'_.txt', 'w') as f:
                f.write(output_txt1)
    quit()