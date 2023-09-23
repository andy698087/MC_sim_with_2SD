#This one, using the multithread, runs quickly (about 10 min for each 100,000 simulations). It obtains results identical to at least 16 digits as the one using no multithread.
import pandas as pd
import numpy as np
from scipy.stats import norm, chi2, weibull_min
from datetime import datetime
from math import sqrt, log, exp, gamma
import dask.dataframe as dd
from scipy.optimize import minimize

class weibull_and_lognorm(object):
    def __init__(self, SampleMeanLogNorm, SampleVarianceLogNorm):

        self.SampleMeanLogNorm = SampleMeanLogNorm
        self.SampleVarianceLogNorm = SampleVarianceLogNorm
        
        # self.initial_shape_parameter = 10
        # self.initial_scale_parameter = 1

        # self.keep_shape_parameter = None
        # self.keep_scale_parameter = None
        self.nIter = 0
        self.dict_WeibullParameter_diff = {'shape_parameter': [], 'scale_parameter': [],  'diff': []}
    def sum_diff_weibull_lognorm(self, params, weight_):
        # print('Main')
        # self.keep_scale_parameter = scale_parameter
        # shape_parameter = minimize_scalar(weibull_and_lognorm(25,1,0.3).sum_diff_weibull_lognorm_ShapeParam,  method = 'brent', bracket = [1,100], tol = 1e-6).x
        self.nIter += 1
        shape_parameter, scale_parameter = abs(params)
        # print(f'shape_parameter: {shape_parameter}, scale_parameter: {scale_parameter}')
        
        TheoryMeanWeibull = scale_parameter * gamma(1 + 1/shape_parameter) # + location_parameter
        TheoryVarWeibull = (scale_parameter ** 2) * (gamma(1 + 2/shape_parameter) - (gamma(1 + 1/shape_parameter)) ** 2)

        # print(f'SampleMeanLogNorm: {self.SampleMeanLogNorm}, SampleVarianceLogNorm: {self.SampleVarianceLogNorm}')
        # print(f'TheoryMeanWeibull: {TheoryMeanWeibull}, TheoryVarWeibull: {TheoryVarWeibull}')

        self.diff = abs(TheoryMeanWeibull-self.SampleMeanLogNorm)+abs(TheoryVarWeibull-self.SampleVarianceLogNorm)
        # print(f'abs diff: {self.diff}')

        self.diff_mean = abs(TheoryMeanWeibull-self.SampleMeanLogNorm)
        self.diff_var = abs(TheoryVarWeibull-self.SampleVarianceLogNorm)
        
        if self.diff < 1e-5 and self.diff_mean < 1e-4 and self.diff_var < 1e-4:
            self.dict_WeibullParameter_diff['shape_parameter'].append(shape_parameter)
            self.dict_WeibullParameter_diff['scale_parameter'].append(scale_parameter)
            self.dict_WeibullParameter_diff['diff'].append(self.diff)
            # self.dict_WeibullParameter_diff['diff_mean'].append(self.diff_mean)
            # self.dict_WeibullParameter_diff['diff_var'].append(self.diff_var)

        return self.loss_func(TheoryMeanWeibull, TheoryVarWeibull, self.SampleMeanLogNorm, self.SampleVarianceLogNorm, weight_)
    
    def loss_func(self, a1, a2, b1, b2, weight_):
        return abs(a1-b1)+ weight_ * abs(a2-b2)
        
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
        self.rMeanLogScale1 = log(1)
        self.rMeanLogScale2 = self.rMeanLogScale1
        
        # Standard deviation in log scale, notation "σ_i" in the manuscript, Equation 1 in the manuscript
        self.rSDLogScale1 = sqrt(log(1 + self.CV1 ** 2)) 
        self.rSDLogScale2 = self.rSDLogScale1

        self.x0_pre = [10,1]

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

    def main_early_dd(self):
        # the pre-determined list of seeds, using number of nMonte
        list_seeds = [i for i in range(self.seed_value, self.seed_value + self.nMonte)] 
        # put the list of seeds into a table (a.k.a DataFrame) with one column named "Seeds"  
        df = pd.DataFrame({'Seeds':list_seeds}) 
        df['rSampleOfRandomsLogNorm'] = df.apply(self.Sample_inv_normal, args=('Seeds',), axis=1).apply(lambda x: [np.exp(item) for item in x])
        df_record = df
        # df = df['rSampleOfRandomsLogNorm'].copy()
        # put the table into dask, a progress that can parallel calculating each rows using multi-thread
        df = dd.from_pandas(df['rSampleOfRandomsLogNorm'], npartitions=35) 
        meta = ('float64', 'float64')
        # calculate sample mean and SD using Mean_SD
        df = df.apply(self.Mean_Var, args = (self.N1, self.N2), meta=meta)
        df_record[['LogNormMean1', 'LogNormVar1', 'LogNormMean2', 'LogNormVar2']] = df.compute().tolist()
        
        df1 = df.apply(self.find_WeibullMeanVar2, args = (0,1), meta=meta)
        df2 = df.apply(self.find_WeibullMeanVar2, args = (2,3), meta=meta)
        
        df_record[['shape_parameter1', 'scale_parameter1']] = df1.compute().tolist()
        df_record[['shape_parameter2', 'scale_parameter2']] = df2.compute().tolist()

        df1 = df1.apply(self.Sample_Weibull, args = (0, 1, self.seed_value,), meta=meta).apply(self.Mean_Var, args = (self.N1,0), meta=meta)
        df2 = df2.apply(self.Sample_Weibull, args = (0, 1, self.seed_value,), meta=meta).apply(self.Mean_Var, args = (self.N2,0), meta=meta)
        
        df_record[['WeibullMean1', 'WeibullVar1']] = df1.compute().tolist()
        df_record[['WeibullMean2', 'WeibullVar2']] = df2.compute().tolist()
        df = dd.from_pandas(df_record[['WeibullMean1', 'WeibullVar1','WeibullMean2', 'WeibullVar2']], npartitions=35)
        
        # df[['WeibullMean1', 'WeibullVar1']] = df1.apply(lambda x: pd.Series(x))
        # df[['WeibullMean2', 'WeibullVar2']] = df2.apply(lambda x: pd.Series(x))

        # df = dd.from_pandas(df[['WeibullMean1', 'WeibullVar1', 'WeibullMean2', 'WeibullVar2']], npartitions=35) 

        # using first_two_moment to transform from raw to log mean and SD                        
        # using Equation 7 and 8
        # generate sample mean and SD in Log scale using Mean_SD
        df = df.apply(self.first_two_moment, args=(0,1,2,3), meta=('float64', 'float64'), axis=1) 
        
        df_record[['rSampleMeanLogScale1', 'rSampleSDLogScale1', 'rSampleMeanLogScale2', 'rSampleSDLogScale2']] = df.compute().tolist()
        
        # Equation 3 # generate 'ln_ratio' and 'se_ln_ratio' with sample mean and SD using GPM
        df = df.apply(self.GPM_log_ratio_SD, args=(0,1,2,3), meta=('float64', 'float64'))  
        df_record[['ln_ratio', 'se_ln_ratio']] = df.compute().tolist()
        # check coverage of each rows
        df = df.apply(self.Coverage, args=(0,1), meta = ('float64', 'float64')) 
        df_record['intervals_include_zero'] = df.compute().tolist()

        # compute the mean of the list of coverage (0 or 1), it equals to the percentage of coverage
        coverage = df.mean().compute() 

        return coverage, df_record, self.nMonte, self.N1, self.CV1
    
    # the main process, method of moments = ['no_moments', 'first_two_moment', 'higher_orders_of_moments']
    def main(self):
        # the pre-determined list of seeds, using number of nMonte
        list_seeds = [i for i in range(self.seed_value, self.seed_value + self.nMonte)] 
        # put the list of seeds into a table (a.k.a DataFrame) with one column named "Seeds"  
        df = pd.DataFrame({'Seeds':list_seeds}) 
        df['rSampleOfRandomsLogNorm'] = df.apply(self.Sample_inv_normal, args=('Seeds',), axis=1).apply(lambda x: [np.exp(item) for item in x])
        df_record = df.copy()
        df = df['rSampleOfRandomsLogNorm'].copy()
        # put the table into dask, a progress that can parallel calculating each rows using multi-thread
        # df = dd.from_pandas(df['rSampleOfRandomsLogNorm'], npartitions=35) 
        # calculate sample mean and SD using Mean_SD
        df = df.apply(self.Mean_Var, args = (self.N1, self.N2))
        df_record[['LogNormMean1', 'LogNormVar1', 'LogNormMean2', 'LogNormVar2']] = df.apply(lambda x: pd.Series(x))
        
        df1 = df.apply(self.find_WeibullMeanVar2, args = (0,1))
        df2 = df.apply(self.find_WeibullMeanVar2, args = (2,3))
        
        df_record[['shape_parameter1', 'scale_parameter1']] = df1.apply(lambda x: pd.Series(x))
        df_record[['shape_parameter2', 'scale_parameter2']] = df2.apply(lambda x: pd.Series(x))

        df1 = df1.apply(self.Sample_Weibull, args = (0, 1, self.seed_value,)).apply(self.Mean_Var, args = (self.N1,0))
        df2 = df2.apply(self.Sample_Weibull, args = (0, 1, self.seed_value,)).apply(self.Mean_Var, args = (self.N2,0))
        
        df_record[['WeibullMean1', 'WeibullVar1']] = df1.apply(lambda x: pd.Series(x))
        df_record[['WeibullMean2', 'WeibullVar2']] = df2.apply(lambda x: pd.Series(x))

        df = pd.DataFrame(df)
        df[['WeibullMean1', 'WeibullVar1']] = df1.apply(lambda x: pd.Series(x))
        df[['WeibullMean2', 'WeibullVar2']] = df2.apply(lambda x: pd.Series(x))
        df = dd.from_pandas(df[['WeibullMean1', 'WeibullVar1', 'WeibullMean2', 'WeibullVar2']], npartitions=35) 
        # using first_two_moment to transform from raw to log mean and SD                        
        # using Equation 7 and 8
        # generate sample mean and SD in Log scale using Mean_SD
        df = df.apply(self.first_two_moment, args=(0,1,2,3), meta=('float64', 'float64'), axis=1) 
        
        df_record[['rSampleMeanLogScale1', 'rSampleSDLogScale1', 'rSampleMeanLogScale2', 'rSampleSDLogScale2']] = df.compute().tolist()
        
        # Equation 3 # generate 'ln_ratio' and 'se_ln_ratio' with sample mean and SD using GPM
        df = df.apply(self.GPM_log_ratio_SD, args=(0,1,2,3), meta=('float64', 'float64'))  
        df_record[['ln_ratio', 'se_ln_ratio']] = df.compute().tolist()
        # check coverage of each rows
        df = df.apply(self.Coverage, args=(0,1), meta = ('float64', 'float64')) 
        df_record['intervals_include_zero'] = df.compute().tolist()

        # compute the mean of the list of coverage (0 or 1), it equals to the percentage of coverage
        coverage = df.mean().compute() 

        return coverage, df_record, self.nMonte, self.N1, self.CV1

    def sum_diff_weibull_lognorm(self, params, weight_, SampleMeanLogNorm, SampleVarianceLogNorm):
        shape_parameter, scale_parameter = abs(params)
        theoryMeanWeibull, theoryVarWeibull = self.theory_Weibull_MeanVar(shape_parameter, scale_parameter)

        return self.loss_func(theoryMeanWeibull, theoryVarWeibull, SampleMeanLogNorm, SampleVarianceLogNorm, weight_)     

    def theory_Weibull_MeanVar(self, shape_parameter, scale_parameter):
        theoryMeanWeibull = scale_parameter * gamma(1 + 1/shape_parameter) # + location_parameter
        theoryVarWeibull = (scale_parameter ** 2) * (gamma(1 + 2/shape_parameter) - (gamma(1 + 1/shape_parameter)) ** 2)

        return theoryMeanWeibull, theoryVarWeibull

    def find_WeibullMeanVar2(self, row, col_SampleMeanLogNorm, col_SampleVarLogNorm):
        nIter = 0
        SampleMeanLogNorm, SampleVarLogNorm = row[col_SampleMeanLogNorm], row[col_SampleVarLogNorm]

        weight_ = 10
        x0 = self.x0_pre
        bounds_ = [(0.5,None),(0.1,None)]
        options_ = {'ftol': 1e-7, 'xtol': 1e-10, 'eta': 0.01/(nIter//100 + 1), 'disp': False}
        
        
        while True:
            nIter += 1
            res = minimize(self.sum_diff_weibull_lognorm, x0, args=(weight_, SampleMeanLogNorm, SampleVarLogNorm), method='TNC', bounds=bounds_, options=options_, jac = '3-point')
            shape_parameter, scale_parameter = res.x[0], res.x[1]
            theoryMeanWeibull, theoryVarWeibull = self.theory_Weibull_MeanVar(shape_parameter, scale_parameter)
            diff = abs(theoryMeanWeibull - SampleMeanLogNorm)+abs(theoryVarWeibull - SampleVarLogNorm)
            diff_mean = abs(theoryMeanWeibull - SampleMeanLogNorm)
            diff_var = abs(theoryVarWeibull - SampleVarLogNorm)
            
            if diff < 1e-5 and diff_mean < 1e-5 and diff_var < 1e-5:
                print('optimized res.x:', res.x)      
                print('diff, diff_mean, diff_var:', diff, diff_mean, diff_var)
                self.x0_pre = res.x                          
                break
            else:
                random_ = (1 + (np.random.randint(1,high=10)-5)/100)
                x0 = [max(0.5, res.x[0] * random_), max(0.5, res.x[1] * random_) ]
                weight_ = abs(diff_var/diff_mean) * random_
        
        return shape_parameter, scale_parameter

    def find_WeibullMeanVar(self, row, col_rSampleMeanLogNorm, col_rSampleVarLogNorm):

        rSampleMeanLogNorm, rSampleVarLogNorm = row[col_rSampleMeanLogNorm], row[col_rSampleVarLogNorm]

        fun = weibull_and_lognorm(rSampleMeanLogNorm, rSampleVarLogNorm)
        weight_ = 10
        x0 = self.x0_pre
        bounds_ = [(0.5,None),(0.1,None)]
        options_ = {'ftol': 1e-7, 'xtol': 1e-10, 'eta': 0.01/(getattr(fun,'nIter')//100 + 1), 'disp': False}
        
        while True:
            res = minimize(fun.sum_diff_weibull_lognorm, x0, args=(weight_, ), method='TNC', bounds=bounds_, options=options_, jac = '3-point')
            if (getattr(fun,'diff') < 1e-5 and getattr(fun,'diff_mean') < 1e-5 and getattr(fun,'diff_var') < 1e-5) or getattr(fun,'nIter') > 1e6 or len(getattr(fun,'dict_WeibullParameter_diff')['diff']) > 1e4:                                
                break
            else:
                random_ = (1 + (np.random.randint(1,high=10)-5)/100)
                x0 = [max(0.5, res.x[0] * random_), max(0.5, res.x[1] * random_) ]
                weight_ = abs(getattr(fun,'diff_var')/getattr(fun,'diff_mean')) * random_
        
        df_WeibullParameter_diff = pd.DataFrame(getattr(fun,'dict_WeibullParameter_diff'))
        print("df_WeibullParameter_diff['diff'].min():", df_WeibullParameter_diff['diff'].min())
        shape_parameter, scale_parameter = df_WeibullParameter_diff.loc[df_WeibullParameter_diff['diff'].idxmin()][['shape_parameter', 'scale_parameter']]
        return shape_parameter, scale_parameter

    def Sample_Weibull(self, row, col_shape_parameter, col_scale_parameter, seed_):
        shape_parameter = row[col_shape_parameter]
        scale_parameter = row[col_scale_parameter]
        rSampleOfRandoms = weibull_min.rvs(shape_parameter, scale=scale_parameter, size=self.N1+self.N2, random_state = seed_)
        
        return rSampleOfRandoms

    def Sample_inv_normal(self, row, seed_):        
        # generate log-normal distribution, using mean of rMeanLogScale and standard deviation of rSDLogScale
        rSampleOfRandoms = norm.rvs(loc=self.rMeanLogScale1, scale=self.rSDLogScale1, size=self.N1+self.N2, random_state = row[seed_])
        
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

    def Mean_Var(self, row, N1, N2):
        
        rSampleOfRandoms1 = row[:N1]
        # the mean of rSampleOfRandoms1, notation "z_i"
        rSampleMean1 = np.mean(rSampleOfRandoms1)  
        # the standard deviation of rSampleOfRandoms1, delta degree of freeden = 1, notation "sz_i"
        rSampleVar1 = np.var(rSampleOfRandoms1, ddof=1) 
        
        if N2 != 0:
            rSampleOfRandoms2 = row[N1:(N1+N2)]
            rSampleMean2 = np.mean(rSampleOfRandoms2)
            rSampleVar2 = np.var(rSampleOfRandoms2, ddof=1)
            return rSampleMean1, rSampleVar1, rSampleMean2, rSampleVar2
        else:
            return rSampleMean1, rSampleVar1
              
    def first_two_moment(self, row, col_SampleMean1, col_SampleVar1, col_SampleMean2, col_SampleVar2):
        
        SampleMean1 = row[col_SampleMean1]
        SampleSD1 = sqrt(row[col_SampleVar1])

        SampleMean2 = row[col_SampleMean2]
        SampleSD2 = sqrt(row[col_SampleVar2])

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
    
    def loss_func(self, a1, a2, b1, b2, weight_):
        return abs(a1-b1)+ weight_ * abs(a2-b2)
        
if __name__ == '__main__':
    # number of Monte Carlo simulations
    nMonteSim = 1
    # Sample size, we choose 15, 25, 50, notation "n" in the manuscript
    for N in [25]:#[15, 25, 50]: 
        # coefficient of variation, we choose 0.15, 0.3, 0.5
        for CV in [0.3]:#[0.15, 0.3, 0.5]: 
            # record the datetime at the start
            start_time = datetime.now() 
            print('start_time:', start_time) 
            print(f"Start GPM_MC_nMonteSim_{nMonteSim}_N_{N}_CV_{CV}_{str(start_time).split('.')[0].replace('-','').replace(' ','').replace(':','')}")

            # Cal the class SimulPivotMC(), generate variables in the def __init__(self)
            run = SimulPivotMC(nMonteSim, N, CV)  
            # start main()
            coverage_by_ln_ratio, df_record, nMonte, N1, CV1 = run.main_early_dd()  
            
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
            
            output_dir = f"GPM_MC_nMonte_{nMonte}_N_{N1}_CV_{CV1}_{str(end_time).split('.')[0].replace('-','').replace(' ','').replace(':','')}"
            
            # save the results to the csv
            print('csv save to ' + output_dir + f'_.csv')
            df_record.to_csv(output_dir + f'_.csv')

            # save the results to the txt
            with open(output_dir + f'_.txt', 'w') as f:
                f.write(output_txt1)
    quit()