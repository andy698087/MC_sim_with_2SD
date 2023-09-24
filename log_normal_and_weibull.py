import numpy as np
import pandas as pd
from scipy.stats import norm, weibull_min
from math import gamma, log, sqrt, exp
from scipy.optimize import minimize, Bounds, rosen_der, least_squares, root_scalar, minimize_scalar
from time import sleep

class weibull_and_lognorm(object):
    def __init__(self, N, CVTimeScale, seed_=21431351):
        self.N = N
        
        MeanLogScale = 0
        print('MeanLogScale:',MeanLogScale)
        SDLogScale = sqrt(log(1 + CVTimeScale ** 2))
        print('SDLogScale:',SDLogScale)
        self.seed_ = seed_
        np.random.seed(seed_)
        SamplesLogNorm = np.exp([(norm.ppf(i,loc=MeanLogScale, scale=SDLogScale)) for i in np.random.rand(self.N)] )
        print('SamplesLogNorm:',SamplesLogNorm)
        self.SampleMeanLogNorm = np.mean(SamplesLogNorm)    
        self.SampleVarianceLogNorm = np.var(SamplesLogNorm, ddof=1)
        
        self.x0_pre = [10,1]
        # self.initial_shape_parameter = 10
        # self.initial_scale_parameter = 1

        # self.keep_shape_parameter = None
        # self.keep_scale_parameter = None
        self.nIter = 0
        self.dict_WeibullParameter_diff = {'shape_parameter': [], 'scale_parameter': [],  'diff': [], 'diff_mean': [], 'diff_var': []}
    def sum_diff_weibull_lognorm(self, params, weight_):
        # print('Main')
        # self.keep_scale_parameter = scale_parameter
        # shape_parameter = minimize_scalar(weibull_and_lognorm(25,1,0.3).sum_diff_weibull_lognorm_ShapeParam,  method = 'brent', bracket = [1,100], tol = 1e-6).x
        self.weight_ = weight_
        self.nIter += 1
        shape_parameter, scale_parameter = abs(params)
        shape_parameter = max(shape_parameter,0.1)
        scale_parameter = max(scale_parameter,0.1)
        print(f'shape_parameter: {shape_parameter}, scale_parameter: {scale_parameter}')
        self.dict_WeibullParameter_diff['shape_parameter'].append(shape_parameter)
        self.dict_WeibullParameter_diff['scale_parameter'].append(scale_parameter)

        SamplesWeibull = weibull_min.rvs(abs(shape_parameter), scale=abs(scale_parameter), size=self.N, random_state = self.seed_)
        SampleMeanWeibull = np.mean(SamplesWeibull)
        SampleVarianceWeibull = np.var(SamplesWeibull, ddof=1)
        print(f'SampleMeanLogNorm: {self.SampleMeanLogNorm}, SampleVarianceLogNorm: {self.SampleVarianceLogNorm}')
        print(f'SampleMeanWeibull: {SampleMeanWeibull}, SampleVarianceWeibull: {SampleVarianceWeibull}')

        self.diff = abs(SampleMeanWeibull-self.SampleMeanLogNorm)+abs(SampleVarianceWeibull-self.SampleVarianceLogNorm)
        self.dict_WeibullParameter_diff['diff'].append(self.diff)
        print(f'diff: {self.diff}')
        
        self.diff_mean = abs(SampleMeanWeibull-self.SampleMeanLogNorm)
        self.diff_var = abs(SampleVarianceWeibull-self.SampleVarianceLogNorm)

        if self.diff < 1e-5 and self.diff_mean < 1e-5 and self.diff_var < 1e-5:
            self.dict_WeibullParameter_diff['shape_parameter'].append(shape_parameter)
            self.dict_WeibullParameter_diff['scale_parameter'].append(scale_parameter)
            self.dict_WeibullParameter_diff['diff'].append(self.diff)
            self.dict_WeibullParameter_diff['diff_mean'].append(self.diff_mean)
            self.dict_WeibullParameter_diff['diff_var'].append(self.diff_var)
        
        return self.loss_func(SampleMeanWeibull, SampleVarianceWeibull, self.SampleMeanLogNorm, self.SampleVarianceLogNorm)/self.nIter
    
    def loss_func(self, a1, a2, b1, b2):
        # print('self.weight_:',self.weight_)
        return abs(a1-b1)+ self.weight_ * abs(a2-b2)
    
    def loss_func2(self, a1, a2, b1, b2):
        return (abs(a1-b1)+ self.weight_ * abs(a2-b2)) +  (-log(a2/b2))
    
    def loss_func3(self, a1, a2, b1, b2):
        # print(a1, b1, a2, b2)
        return log(exp(abs(a1-b1))+ exp( abs(a2-b2))) +  (-log(a2/b2))

    
    def sum_diff_weibull_lognorm2(self, params):
        shape_parameter, scale_parameter = abs(params)

        SamplesWeibull = weibull_min.rvs(abs(shape_parameter), scale=abs(scale_parameter), size=self.N, random_state = self.seed_)
        self.SampleMeanWeibull = np.mean(SamplesWeibull)
        self.SampleVarianceWeibull = np.var(SamplesWeibull, ddof=1)

        return self.loss_func(self.SampleMeanWeibull, self.SampleVarianceWeibull, self.SampleMeanLogNorm, self.SampleVarianceLogNorm)     

    def diff_out(self):
        return self.diff, self.diff_mean, self.diff_var
    
    def find_WeibullMeanVar2(self):
        nIter = 0

        self.weight_ = 10
        x0 = self.x0_pre
        bounds_ = [(0.5,None),(0.1,None)]
        options_ = {'ftol': 1e-7, 'xtol': 1e-10, 'eta': 0.01/(nIter//100 + 1), 'disp': False}
        
        
        while True:
            nIter += 1
            res = minimize(self.sum_diff_weibull_lognorm2, x0, method='TNC', bounds=bounds_, options=options_, jac = '3-point')
            shape_parameter, scale_parameter = res.x
            # theoryMeanWeibull, theoryVarWeibull = theory_Weibull_MeanVar(shape_parameter, scale_parameter)
            
            diff = abs(self.SampleMeanWeibull - self.SampleMeanLogNorm)+abs(self.SampleVarianceWeibull - self.SampleVarianceLogNorm)
            diff_mean = abs(self.SampleMeanWeibull - self.SampleMeanLogNorm)
            diff_var = abs(self.SampleVarianceWeibull - self.SampleVarianceLogNorm)
            print('diff:',diff)
            if diff < 1e-6 and diff_mean < 1e-5 and diff_var < 1e-5:
                print('optimized res.x:', res.x)      
                print('diff, diff_mean, diff_var:', diff, diff_mean, diff_var)
                print("SampleMeanWeibull, SampleMeanLogNorm:", self.SampleMeanWeibull, self.SampleMeanLogNorm)
                print("SampleVarianceWeibull, SampleVarianceLogNorm:", self.SampleVarianceWeibull, self.SampleVarianceLogNorm)
                
                shape_parameter, scale_parameter = res.x 
                self.x0_pre = res.x                          
                break
            else:
                random_ = (1 + (np.random.randint(1,high=10)-5)/100)
                x0 = [max(0.5, res.x[0] * random_), max(0.5, res.x[1] * random_) ]
                weight_  = abs(diff_var/diff_mean) * random_

        return shape_parameter, scale_parameter


fun = weibull_and_lognorm(15, 0.5)
fun.find_WeibullMeanVar2()

quit()

methods = ['Nelder-Mead','Powell','CG', 'TNC']
fun = weibull_and_lognorm(25,1,0.3)
weight_ = 10



x0 = [10,1]
while True:
    
    bounds_ = [(0.5,None),(0.1,None)]
    options_ = {'ftol': 1e-7, 'xtol': 1e-20, 'eta': 0.1, 'disp': True}

    res = minimize(fun.sum_diff_weibull_lognorm, x0, args=(weight_, ), method='TNC', bounds=bounds_, options=options_, jac = '3-point')
    # res = least_squares(weibull_and_lognorm(25,1,0.3).sum_diff_weibull_lognorm, x0,)
    # res = minimize_scalar(weibull_and_lognorm(25,1,0.3).sum_diff_weibull_lognorm,  method = 'brent', bracket = [1,100], tol = 1e-6)
    print(f'res.x: {res.x}')
    if fun.diff_out()[0] < 1e-6:
        break
    else:
        random_ = (1 + (np.random.randint(1,high=10)-5)/100)
        x0 = [res.x[0] * random_, res.x[1] * random_ ]
        weight_ = min(1000, abs(getattr(fun,'diff_var')/getattr(fun,'diff_mean')) * random_)
        print(f"diff_mean: {getattr(fun,'diff_mean')}, diff_var: {getattr(fun,'diff_var')}")
        print(f'random: {random_}')
        print(f'weight_:{weight_}')
        print(f'new x0: {x0}')
        print(f'nIter: {fun.nIter}')
        
        pd.DataFrame(getattr(fun,'dict_WeibullParameter_diff')).to_csv('WeibullParameter_diff.csv')
        sleep(1)

quit()

# 设置Weibull分布的参数（根据所选参数化方式）
shape_parameter = 2.0  # 形状参数(c)
scale_parameter = 3.0  # 尺度参数(λ或scale)

# 计算理论均值和方差
theory_mean = scale_parameter * gamma(1 + 1/shape_parameter) # + location_parameter
theory_variance = (scale_parameter ** 2) * (gamma(1 + 2/shape_parameter) - (gamma(1 + 1/shape_parameter)) ** 2)

# 生成模拟数据
num_samples = 30  # 模拟样本数量
simulated_data = stats.weibull_min.rvs(shape_parameter, loc=0, scale=scale_parameter, size=num_samples)

# 计算模拟数据的样本均值和样本方差
sample_mean = np.mean(simulated_data)
sample_variance = np.var(simulated_data)

# 输出结果
print("理论均值:", theory_mean)
print("理论方差:", theory_variance)
print("模拟样本均值:", sample_mean)
print("模拟样本方差:", sample_variance)





import numpy as np

# Define the parameters of the log-normal distribution on the logarithmic scale
mu_log = 0.0  # Mean on the logarithmic scale
sigma_log = 1.0  # Standard deviation on the logarithmic scale

# Calculate the mean and variance on the time scale
mean_time_scale = np.exp(mu_log + (sigma_log ** 2) / 2)
variance_time_scale = (np.exp(sigma_log ** 2) - 1) * np.exp(2 * mu_log + sigma_log ** 2)

print("Mean on Time Scale:", mean_time_scale)
print("Variance on Time Scale:", variance_time_scale)
