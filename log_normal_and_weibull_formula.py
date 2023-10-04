import numpy as np
import pandas as pd
from scipy.stats import norm, weibull_min
from math import gamma, log, sqrt, exp
from scipy.optimize import minimize, Bounds, rosen_der, least_squares, root_scalar, minimize_scalar
from time import sleep
from datetime import datetime

class weibull_and_lognorm(object):

    def __init__(self, N, CVTimeScale, seed_=20230922):
        self.N = N
        
        MeanLogScale = 0
        # MeanLogScale_1 = log(Mean/sqrt(CVsq + 1)) 
        # self.MeanTimeScale = exp(MeanLogScale) * sqrt(CV**2 +1)
        self.MeanTimeScale = 1
        self.VarTimeScale = (CVTimeScale * self.MeanTimeScale)**2
        # self.VarLogScale = log(1 + CVTimeScale ** 2)
        print('self.MeanTimeScale:',self.MeanTimeScale)
        print('self.VarTimeScale:',self.VarTimeScale)
        sleep(2)
        # SDLogScale = sqrt(log(1 + CVTimeScale ** 2))
        self.seed_value = seed_
        # np.random.seed(seed_)
        # SamplesLogNorm = np.exp([(norm.ppf(i,loc=MeanLogScale, scale=SDLogScale)) for i in np.random.rand(self.N)] )
        
        # self.SampleMeanLogNorm = np.mean(SamplesLogNorm)    
        # self.SampleVarianceLogNorm = np.var(SamplesLogNorm)
        
        # self.initial_shape_parameter = 10
        # self.initial_scale_parameter = 1

        # self.keep_shape_parameter = None
        # self.keep_scale_parameter = None
        self.x0_pre = [10,1]
        self.weight_ = 10
    def sum_diff_weibull_lognorm(self, params, MeanTimeScale, VarTimeScale):
        shape_parameter, scale_parameter = abs(params)
        
        MeanWeibull, VarWeibull = self.theory_Weibull_MeanVar(shape_parameter, scale_parameter)
        
        diff = abs(MeanWeibull - MeanTimeScale)+abs(VarWeibull - VarTimeScale)
        diff_mean = abs(MeanWeibull - MeanTimeScale)
        diff_var = abs(VarWeibull - VarTimeScale)
        # print('diff:',diff)
        # print('MeanWeibull:',MeanWeibull)
        # print('VarWeibull:',VarWeibull)
        if diff < 1e-6 and diff_mean < 1e-6 and diff_var < 1e-6:
            self.dict_WeibullParameter_diff['MeanTimeScale'].append(MeanTimeScale)
            self.dict_WeibullParameter_diff['VarTimeScale'].append(VarTimeScale)
            self.dict_WeibullParameter_diff['shape_parameter'].append(shape_parameter)
            self.dict_WeibullParameter_diff['scale_parameter'].append(scale_parameter)
            self.dict_WeibullParameter_diff['MeanWeibull'].append(MeanWeibull)
            self.dict_WeibullParameter_diff['VarWeibull'].append(VarWeibull)
            self.dict_WeibullParameter_diff['diff'].append(diff)
            self.dict_WeibullParameter_diff['diff_mean'].append(diff_mean)
            self.dict_WeibullParameter_diff['diff_var'].append(diff_var)
            # print('diff < 1e-5')
            # print(self.dict_WeibullParameter_diff)
            # sleep(1)
        
        return self.loss_func(MeanWeibull, VarWeibull, MeanTimeScale, VarTimeScale)     
    
    def theory_Weibull_MeanVar(self, shape_parameter, scale_parameter):
        theoryMeanWeibull = scale_parameter * gamma(1 + 1/shape_parameter) # + location_parameter
        theoryVarWeibull = (scale_parameter ** 2) * (gamma(1 + 2/shape_parameter) - (gamma(1 + 1/shape_parameter)) ** 2)

        return theoryMeanWeibull, theoryVarWeibull

    def find_WeibullMeanVar(self):
        nIter = 0
        x0 = self.x0_pre
        self.dict_WeibullParameter_diff = {'MeanTimeScale': [], 'VarTimeScale': [], 'shape_parameter': [], 'scale_parameter': [], 'MeanWeibull': [], 'VarWeibull': [], 'diff': [], 'diff_mean': [], 'diff_var': []}
        bounds_ = [(0.5,None),(0.1,None)]
        options_ = {'ftol': 1e-8, 'xtol': 1e-10, 'eta': 0.01/(nIter//1000 + 1), 'disp': False}
        
        # print('start find weibull')
        while True:
            nIter += 1
            res = minimize(self.sum_diff_weibull_lognorm, x0, args=(self.MeanTimeScale, self.VarTimeScale), method='TNC', bounds=bounds_, options=options_, jac = '3-point')
            # shape_parameter, scale_parameter = res.x

            # diff = abs(self.MeanWeibull - SampleMeanLogNorm)+abs(self.VarWeibull - SampleVarLogNorm)
            # diff_mean = abs(self.MeanWeibull - SampleMeanLogNorm)
            # diff_var = abs(self.VarWeibull - SampleVarLogNorm)
            # # print('diff:',diff)
            # print('find_WeibullMeanVar self.dict_WeibullParameter_diff:',self.dict_WeibullParameter_diff)
            
            # print('extract_df:',extract_df)
            # sleep(1)
            # extract_df = extract_df[(extract_df['SampleMeanLogNorm'] == SampleMeanLogNorm) & (extract_df['SampleVarianceLogNorm'] == SampleVarLogNorm)]
            if len(self.dict_WeibullParameter_diff['diff']) > 0:
                extract_df = pd.DataFrame(self.dict_WeibullParameter_diff)
                extract_df = extract_df.loc[extract_df['diff'].idxmin()]
                # print('extract_df:',extract_df)
                diff = extract_df['diff']
                # diff_mean = extract_df['diff_mean']
                # diff_var = extract_df['diff_var']
                if diff < 1e-6:
                    # print('optimized res.x:', res.x)      
                    # print(pd.DataFrame(extract_df))
                    self.extract_df = extract_df
                    shape_parameter, scale_parameter = res.x 
                    print('extract_df', extract_df)
                    # shape_parameter, scale_parameter = extract_df[['shape_parameter','scale_parameter']]
                    # self.x0_pre = res.x                          
                    break
            else:
                np.random.seed(self.seed_value+nIter)
                random_ = (1 + (np.random.randint(1,high=10)-5)/100)
                x0 = [max(0.5, res.x[0] * random_), max(0.5, res.x[1] * random_) ]
                # print('x0:', x0)
                # self.weight_ = abs(extract_df['diff_var']/extract_df['diff_mean']) * random_
                # print('self.weight_:',self.weight_)
                # sleep(2)

        return shape_parameter, scale_parameter
    
    def loss_func(self, a1, a2, b1, b2):
        return abs(a1-b1)+  abs(a2-b2)
    
    def loss_func2(self, a1, a2, b1, b2):
        return (abs(a1-b1)+ weight_ * abs(a2-b2)) +  (-log(a2/b2))
    
    def loss_func3(self, a1, a2, b1, b2):
        # print(a1, b1, a2, b2)
        return log(exp(abs(a1-b1))+  exp(weight_ *  abs(a2-b2))) +  (-log(a2/b2))
    

weibull_params = {'N':[],'CV':[],'shape_parameter':[],'scale_parameter':[],'MeanTimeScale': [], 'VarTimeScale': [], 'shape_parameter': [], 'scale_parameter': [], 'MeanWeibull': [], 'VarWeibull': [], 'diff': [], 'diff_mean': [], 'diff_var': []}
for N in [25]: 
    # coefficient of variation, we choose 0.15, 0.3, 0.5
    for CV in [0.15, 0.20, 0.25]: 
        print('N,CV:',N,CV)
        weibull_params['N'].append(N)
        weibull_params['CV'].append(CV)
        fun = weibull_and_lognorm(N, CV)
        shape_parameter, scale_parameter = fun.find_WeibullMeanVar()
        weibull_params['shape_parameter'].append(shape_parameter)
        weibull_params['scale_parameter'].append(scale_parameter)
        extract_df = getattr(fun, 'extract_df')
        weibull_params['MeanTimeScale'].append(extract_df['MeanTimeScale'])
        weibull_params['VarTimeScale'].append(extract_df['VarTimeScale'])
        weibull_params['MeanWeibull'].append(extract_df['MeanWeibull'])
        weibull_params['VarWeibull'].append(extract_df['VarWeibull'])
        weibull_params['diff'].append(extract_df['diff'])
        weibull_params['diff_mean'].append(extract_df['diff_mean'])
        weibull_params['diff_var'].append(extract_df['diff_var'])
pd.DataFrame(weibull_params).to_csv('weibull_params1e-6_mean_1_CV_25_50_1_2.csv')
quit()


methods = ['Nelder-Mead','Powell','CG', 'TNC']

for N in [15]:
    for CV in [0.15, 0.3, 0.5]:

        fun = weibull_and_lognorm(N,1,CV)
        weight_ = 10

        x0 = [10,1]

        while True:
            
            bounds_ = [(0.5,None),(0.1,None)]
            options_ = {'ftol': 1e-7, 'xtol': 1e-10, 'eta': 0.01/(getattr(fun,'nIter')//100 + 1), 'disp': True}

            res = minimize(fun.sum_diff_weibull_lognorm, x0, args=(weight_, ), method='TNC', bounds=bounds_, options=options_, jac = '3-point')
            # res = least_squares(weibull_and_lognorm(25,1,0.3).sum_diff_weibull_lognorm, x0,)
            # res = minimize_scalar(weibull_and_lognorm(25,1,0.3).sum_diff_weibull_lognorm,  method = 'brent', bracket = [1,100], tol = 1e-6)
            print(f'res.x: {res.x}')
            if getattr(fun,'diff') < 1e-6 or getattr(fun,'nIter') > 1e6 or len(getattr(fun,'dict_WeibullParameter_diff')['diff']) > 1e3:
                end_time = str(datetime.now()).split('.')[0].replace('-','').replace(' ','').replace(':','')
                pd.DataFrame(getattr(fun,'dict_WeibullParameter_diff')).to_csv(f'WeibullParameter_diff_theory_{N}_{CV}_{end_time}.csv')
                break
            else:
                random_ = (1 + (np.random.randint(1,high=10)-5)/100)
                x0 = [max(0.5, res.x[0] * random_), max(0.5, res.x[1] * random_) ]
                weight_ = abs(getattr(fun,'diff_var')/getattr(fun,'diff_mean')) * random_
                print(f"diff_mean: {getattr(fun,'diff_mean')}, diff_var: {getattr(fun,'diff_var')}")
                print(f'random: {random_}')
                print(f'weight_:{weight_}')
                print(f'new x0: {x0}')
                print(f'nIter: {fun.nIter}')
                pd.DataFrame(getattr(fun,'dict_WeibullParameter_diff')).to_csv(f'WeibullParameter_diff_theory_{N}_{CV}_temp.csv')
                sleep(1)

quit()