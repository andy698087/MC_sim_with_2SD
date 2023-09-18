#There are two code files. This one, using no multithread, runs slowly, but easier to read and obtains results identical to at least 16 digits as the multithread used for the simulations.
import pandas as pd
import numpy as np
from scipy.stats import norm, chi2
from datetime import datetime
from math import sqrt, log, exp


def Pivot_calculation(rSampleMeanLogScale, rSampleSDLogScale, N, U, Z):
    # Equation 3
    return np.exp(rSampleMeanLogScale- np.sqrt((rSampleSDLogScale**2 * (N-1))/U) * (Z/sqrt(N)) ) * np.sqrt((np.exp((rSampleSDLogScale**2 * (N-1))/U) - 1) * np.exp((rSampleSDLogScale**2 * (N-1))/U))

def transform_from_raw_to_log_mean_SD(Mean, SD):
    CV = SD/Mean
    CVsq = CV**2
    #Mean in log scale, Equation 7
    MeanLogScale_1 = log(Mean/sqrt(1 + CVsq)) 
    #SD in log scale, Equation 8
    SDLogScale_1 = sqrt(log((CVsq + 1)))
    SDLogScale_2 = sqrt(CVsq * (1 + CVsq/(1 + CVsq))**2 - (1 + CVsq/(1 + CVsq)) * ((1 + CVsq)**2 - 3 + 2/(1 + CVsq)) + (1/4) * ((1 + CVsq)**4 - 4*(1 + CVsq) - 1 + 8/(1 + CVsq) - 4/((1 + CVsq)**2))
)   #SD in log scale, Equation 9
    return MeanLogScale_1, SDLogScale_1, SDLogScale_2


# number of Monte Carlo Simulations
nMonte = 1

# Calculate z-score for alpha = 0.05, 
# ppf is the percent point function that is inverse of cumulative distribution function
z_score = norm.ppf(1 - 0.05 / 2)        

# the number for pivot, the notation "m" in the manuscript
nSimulForPivot = 100000-1

# choosing a seed
seed_value = 12181988

# Generate 4 set of random numbers each with specified seed, will be used for U1, U2, Z1, and Z2 later (Equation 3)
np.random.seed(seed_value - 1)
random_numbers1_1 = np.random.rand(nSimulForPivot)

np.random.seed(seed_value - 2)
random_numbers1_2 = np.random.rand(nSimulForPivot)

np.random.seed(seed_value - 3)
random_numbers2_1 = np.random.rand(nSimulForPivot)

np.random.seed(seed_value - 4)
random_numbers2_2 = np.random.rand(nSimulForPivot)

# for Table 1, 2, and 3, respectively
for MoM in ['no_moments', 'first_two_moment', 'higher_orders_of_moments']:

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
            dict_results = {'ln_ratio': [], 'se_ln_ratio': [], 'coverage': []}
            # the pre-determined list of seeds, using number of nMonte
            list_seeds = [i for i in range(seed_value, seed_value + nMonte)] 
            for seed_ in list_seeds:
                # Calculate the mean and standard deviation of a sample generated from a random generator of a normal distribution
                np.random.seed(seed_)
                # generate log-normal distribution, using mean of rMeanLogScale and standard deviation of rSDLogScale
                rSampleOfRandoms = [norm.ppf(i,loc=rMeanLogScale1, scale=rSDLogScale1) for i in np.random.rand(N1+N2)]
                dict_results['rSampleOfRandoms'].append(np.exp(rSampleOfRandoms))
                #using no method of moments 
                if method_of_moments == 'no_moments': 
                    rSampleOfRandoms1 = rSampleOfRandoms[:N1]
                    rSampleOfRandoms2 = rSampleOfRandoms[N1:N1+N2] 
                    # the mean of rSampleOfRandoms1, notation "z_i"
                    rSampleMeanLogScale1 = np.mean(rSampleOfRandoms1) 
                    # the standard deviation of rSampleOfRandoms1, delta degree of freeden = 1, notation "sz_i"
                    rSampleSDLogScale1 = np.std(rSampleOfRandoms1, ddof=1) 
                    rSampleMeanLogScale2 = np.mean(rSampleOfRandoms2)
                    rSampleSDLogScale2 = np.std(rSampleOfRandoms2, ddof=1)

                # using method of moments to transform from raw to log mean and SD
                else:
                    rSampleOfRandoms = np.exp(rSampleOfRandoms)
                    rSampleOfRandoms1 = rSampleOfRandoms[:N1]
                    rSampleOfRandoms2 = rSampleOfRandoms[N1:N1+N2] 

                    rSampleMeanTimeScale1 = np.mean(rSampleOfRandoms1)
                    rSampleSDTimeScale1 = np.std(rSampleOfRandoms1, ddof=1)
                    rSampleMeanTimeScale2 = np.mean(rSampleOfRandoms2)
                    rSampleSDTimeScale2 = np.std(rSampleOfRandoms2, ddof=1)

                    #using Equation 7 and 8
                    if method_of_moments == 'first_two_moment': 
                        rSampleMeanLogScale1, rSampleSDLogScale1, _ = transform_from_raw_to_log_mean_SD(rSampleMeanTimeScale1, rSampleSDTimeScale1)
                        rSampleMeanLogScale2, rSampleSDLogScale2, _ = transform_from_raw_to_log_mean_SD(rSampleMeanTimeScale2, rSampleSDTimeScale2)
                    
                    #using Equation 7 and 9
                    elif method_of_moments == 'higher_orders_of_moments': 
                        rSampleMeanLogScale1, _, rSampleSDLogScale1, = transform_from_raw_to_log_mean_SD(rSampleMeanTimeScale1, rSampleSDTimeScale1)
                        rSampleMeanLogScale2, _, rSampleSDLogScale2, = transform_from_raw_to_log_mean_SD(rSampleMeanTimeScale2, rSampleSDTimeScale2)

                # Equation 3 
                Pivot1 = Pivot_calculation(rSampleMeanLogScale1, rSampleSDLogScale1, N1, U1, Z1)
                Pivot2 = Pivot_calculation(rSampleMeanLogScale2, rSampleSDLogScale2, N2, U2, Z2)
                
                # Equation 2, generalized pivotal statistics
                pivot_statistics = np.log(Pivot1) - np.log(Pivot2)

                # Calculate ln ratio and SE ln ratio by percentile and Z statistics
                ln_ratio = pd.Series(pivot_statistics).quantile(.5)
                se_ln_ratio = (pd.Series(pivot_statistics).quantile(.75) - pd.Series(pivot_statistics).quantile(.25))/(norm.ppf(.75) - norm.ppf(.25))
                # Calculate the confidence intervals with z_score
                lower_bound = ln_ratio - z_score * se_ln_ratio
                upper_bound = ln_ratio + z_score * se_ln_ratio   
                
                dict_results['ln_ratio'].append(ln_ratio)
                dict_results['se_ln_ratio'].append(se_ln_ratio)
                dict_results['coverage'].append((lower_bound <= 0) and (upper_bound >= 0))
            
            end_time = datetime.now()
            
            # print out the percentage of coverage
            print(f'MoM={method_of_moments} N={N1} CV={CV1} percentage coverage: {np.mean(dict_results["coverage"])}') 
            
            output_dir = f"GPM_MC_nMonte_{nMonte}_N_{N1}_CV_{CV1}_{str(end_time).split('.')[0].replace('-','').replace(' ','').replace(':','')}"
            print('csv save to ' + output_dir + f'_{method_of_moments}.csv')

            # save the results to the csv
            pd.DataFrame(dict_results).to_csv(output_dir + f'_{method_of_moments}.csv')

