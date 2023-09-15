import pandas as pd
import numpy as np
from scipy.stats import norm, chi2
from datetime import datetime
from math import sqrt, log, exp

# Equation 3
def Pivot_calculation(rSampleMeanLogScale, rSampleSDLogScale, N, U, Z):
    return np.exp(rSampleMeanLogScale- np.sqrt((rSampleSDLogScale**2 * (N-1))/U) * (Z/sqrt(N)) ) *\
                  np.sqrt((np.exp((rSampleSDLogScale**2 * (N-1))/U) - 1) * np.exp((rSampleSDLogScale**2 * (N-1))/U))

#Equation 7, 8, 9 and 10
def transform_from_raw_to_log_mean_SD(N, Mean, SD):
    MeanLogScale_1 = log(Mean) - (1/2) * log(1 + (SD/Mean)**2)   #Mean in log scale, Equation 7
    SDLogScale_1 = sqrt(log((SD/Mean)**2 + 1))   #SD in log scale, Equation 8
    var_x = (SD**2) / N
    dz_dmean = (1/Mean) + (SD**2) / (Mean * ((SD**2) + (Mean**2)))
    cov_x_sx2 = (1/N) * exp(3*MeanLogScale_1) * (exp(9*(SDLogScale_1**2)/2) - 3*exp(5*(SDLogScale_1**2)/2) + 2*exp(3*(SDLogScale_1**2)/2))
    dz_dsx2 = -1 / (2 * ((SD**2) + (Mean**2)))
    var_sx2 = (1/N) * exp(4*MeanLogScale_1) * (exp(8*SDLogScale_1**2) - 4*exp(5*SDLogScale_1**2) - exp(4*SDLogScale_1**2) + 8*exp(3*SDLogScale_1**2) - 4*exp(2*SDLogScale_1**2))
    var_B_z = var_x * dz_dmean**2 + 2 * cov_x_sx2 * dz_dmean * dz_dsx2 + var_sx2 * dz_dsx2**2
    SDLogScale_2 = sqrt(N * var_B_z)   #SD in log scale, Equation 9
    return MeanLogScale_1, SDLogScale_1, SDLogScale_2

# number of Monte Carlo Simulations
nMonte = 100000

# Calculate z-score for alpha = 0.05, 
# ppf is the percent point function that is inverse of cumulative distribution function
z_score = norm.ppf(1 - 0.05 / 2)        

# choosing a seed
seed_value = 12181988

# the number for pivot, the notation "m" in the manuscript
nSimulForPivot = 100000-1

# Generate 4 set of random numbers each with specified seed
np.random.seed(seed_value - 1)
random_numbers1_1 = np.random.rand(nSimulForPivot)

np.random.seed(seed_value - 2)
random_numbers1_2 = np.random.rand(nSimulForPivot)

np.random.seed(seed_value - 3)
random_numbers2_1 = np.random.rand(nSimulForPivot)

np.random.seed(seed_value - 4)
random_numbers2_2 = np.random.rand(nSimulForPivot)

# Method of moments, for Table 1, 2, and 3, respectively
for method_of_moments in ['no_moments', 'first_two_moment', 'higher_orders_of_moments']:

    # Sample size, we choose 15, 25, 50, notation "n" in the manuscript
    for N in [15, 25, 50]:
        N1 = N
        N2 = N1

        for CV in [0.15, 0.3, 0.5]:
            # coefficient of variation, we choose 0.15, 0.3, 0.5
            CV1 = CV
            CV2 = CV1

            # Mean in log scale, notation "μ_i" in the manuscript
            rMeanLogScale1 = 1
            rMeanLogScale2 = rMeanLogScale1
                    
            # Standard deviation in log scale, notation "σ_i" in the manuscript
            rSDLogScale1 = sqrt(log(1 + CV1 ** 2))  #Equation 1 in the manuscript
            rSDLogScale2 = rSDLogScale1

            # number to do the simulation for pivot, notation "m" in the manuscript
            nSimulForPivot = 100000-1

            # Choose the seed
            seed_value = 12181988

            # Generate random number for later used in calculating Ui and Zi in generalized pivotal method
            # group 1 pivot calculation
            # U_i and Z_i used in Equation 3
            U1 = chi2.ppf(random_numbers1_1, N1 - 1 )
            Z1 = norm.ppf(random_numbers2_1)

            #group 2 pivot calculation
            # U_i and Z_i used in Equation 3
            U2 = chi2.ppf(random_numbers1_2, N2 - 1 )
            Z2 = norm.ppf(random_numbers2_2)

            #collecting results
            dict_results = {'seed_':[],'rSampleOfRandoms': [], 'rSampleMeanLogScale1': [], 'rSampleSDLogScale1': [], 'ln_ratio': [], 'se_ln_ratio': [], 'coverage': []}
            # the pre-determined list of seeds, using number of nMonte
            list_seeds = [i for i in range(seed_value, seed_value + nMonte)] 
            for seed_ in list_seeds:
                dict_results['seed_'].append(seed_)
                # calculate the mean and std of a sample generate from random generater of normal distribution
                np.random.seed(seed_)
                # generate log-normal distribution, using mean of rMeanLogScale and standard deviation of rSDLogScale
                rSampleOfRandoms = [norm.ppf(i,loc=rMeanLogScale1, scale=rSDLogScale1) for i in np.random.rand(N1+N2)]
                dict_results['rSampleOfRandoms'].append(np.exp(rSampleOfRandoms))
                if method_of_moments == 'no_moments': #using no method of moments 
                    rSampleOfRandoms1 = rSampleOfRandoms[:N1]
                    rSampleOfRandoms2 = rSampleOfRandoms[N1:N1+N2] 
                    
                    rSampleMeanLogScale1 = np.mean(rSampleOfRandoms1) # the mean of rSampleOfRandoms1, notation "z_i"
                    rSampleSDLogScale1 = np.std(rSampleOfRandoms1, ddof=1) # the standard deviation of rSampleOfRandoms1, delta degree of freeden = 1, notation "sz_i"
                    rSampleMeanLogScale2 = np.mean(rSampleOfRandoms2)
                    rSampleSDLogScale2 = np.std(rSampleOfRandoms2, ddof=1)

                    dict_results['rSampleMeanLogScale1'].append(rSampleMeanLogScale1)
                    dict_results['rSampleSDLogScale1'].append(rSampleSDLogScale1)
                else:# using method of moments to transform from raw to log mean and SD
                    # generate log-normal distribution, using mean of rMeanLogScale and standard deviation of rSDLogScale
                    rSampleOfRandoms = np.exp(rSampleOfRandoms)
                    rSampleOfRandoms1 = rSampleOfRandoms[:N1]
                    rSampleOfRandoms2 = rSampleOfRandoms[N1:N1+N2] 

                    rSampleMeanTimeScale1 = np.mean(rSampleOfRandoms1) # the mean of rSampleOfRandoms1, notation "z_i"
                    rSampleSDTimeScale1 = np.std(rSampleOfRandoms1, ddof=1) # the standard deviation of rSampleOfRandoms1, delta degree of freeden = 1, notation "sz_i"
                    rSampleMeanTimeScale2 = np.mean(rSampleOfRandoms2)
                    rSampleSDTimeScale2 = np.std(rSampleOfRandoms2, ddof=1)
                    
                    if method_of_moments == 'first_two_moment': #using Equation 7 and 8
                        rSampleMeanLogScale1, rSampleSDLogScale1, _ = transform_from_raw_to_log_mean_SD(N1, rSampleMeanTimeScale1, rSampleSDTimeScale1)
                        rSampleMeanLogScale2, rSampleSDLogScale2, _ = transform_from_raw_to_log_mean_SD(N2, rSampleMeanTimeScale2, rSampleSDTimeScale2)
                    elif method_of_moments == 'higher_orders_of_moments': #using Equation 7 and 9
                        rSampleMeanLogScale1, _, rSampleSDLogScale1, = transform_from_raw_to_log_mean_SD(N1, rSampleMeanTimeScale1, rSampleSDTimeScale1)
                        rSampleMeanLogScale2, _, rSampleSDLogScale2, = transform_from_raw_to_log_mean_SD(N2, rSampleMeanTimeScale2, rSampleSDTimeScale2)
                    dict_results['rSampleMeanLogScale1'].append(rSampleMeanLogScale1)
                    dict_results['rSampleSDLogScale1'].append(rSampleSDLogScale1)
                
                # Equation 3 
                Pivot1 = Pivot_calculation(rSampleMeanLogScale1, rSampleSDLogScale1, N1, U1, Z1)
                Pivot2 = Pivot_calculation(rSampleMeanLogScale2, rSampleSDLogScale2, N2, U2, Z2)
                # Equation 2, generalized pivotal statistics
                pivot_statistics = np.log(Pivot1) - np.log(Pivot2)

                # Calculate ln ratio and SE ln ratio by percentile and Z statistics, Equation 4 and 5
                ln_ratio = pd.Series(pivot_statistics).quantile(.5)
                se_ln_ratio = (pd.Series(pivot_statistics).quantile(.75) - pd.Series(pivot_statistics).quantile(.25))/(norm.ppf(.75) - norm.ppf(.25))
                
                # Calculate the confidence intervals with z_score of alpha = 0.05, Equation 6
                lower_bound = ln_ratio - z_score * se_ln_ratio
                upper_bound = ln_ratio + z_score * se_ln_ratio   
                
                dict_results['ln_ratio'].append(ln_ratio)
                dict_results['se_ln_ratio'].append(se_ln_ratio)
                dict_results['coverage'].append(int((lower_bound <= 0) and (upper_bound >= 0)))
            
            end_time = datetime.now()
            print(f'MoM={method_of_moments} N={N1} CV={CV1} percentage coverage: {np.mean(dict_results["coverage"])}') # print out the percentage of coverage
            
            
            output_dir = f"GPM_MC_nMonte_{nMonte}_N_{N1}_CV_{CV1}_{str(end_time).split('.')[0].replace('-','').replace(' ','').replace(':','')}"
            print('csv save to ' + output_dir + f'_{method_of_moments}.csv')
            pd.DataFrame(dict_results).to_csv(output_dir + f'_{method_of_moments}.csv')

