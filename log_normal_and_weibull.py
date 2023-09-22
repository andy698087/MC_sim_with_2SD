import numpy as np
import scipy.stats as stats

# 设置Weibull分布的参数（根据所选参数化方式）
shape_parameter = 2.0  # 形状参数(c)
scale_parameter = 3.0  # 尺度参数(λ或scale)

# 计算理论均值和方差
theory_mean = scale_parameter * np.gamma(1 + 1/shape_parameter)
theory_variance = (scale_parameter ** 2) * (np.gamma(1 + 2/shape_parameter) - (np.gamma(1 + 1/shape_parameter)) ** 2)

# 生成模拟数据
num_samples = 1000  # 模拟样本数量
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
