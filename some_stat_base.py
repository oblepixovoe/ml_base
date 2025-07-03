import math
from scipy.stats import t, norm

def sample_mean(data):
    # Выборочное среднее
    return sum(data) / len(data)

def sample_variance(data):
    # Выборочная дисперсия (несмещённая оценка):
    n = len(data)
    mean = sample_mean(data)
    return sum((x - mean) ** 2 for x in data) / (n - 1)

def sample_std(data):
    # Выборочное стандартное отклонение: s = √s²
    return math.sqrt(sample_variance(data))

def standard_error(data):
    # Стандартная ошибка среднего: SE = s / √n
    return sample_std(data) / math.sqrt(len(data))

def z_score(x, mu, sigma):
    # Z-оценка:
    return (x - mu) / sigma

def t_score(sample_mean_value, mu, sample_std_value, n):
    # T-оценка для среднего:
    se = sample_std_value / math.sqrt(n)
    return (sample_mean_value - mu) / se

def ci_mean_known_sigma(sample_mean_value, sigma, n, alpha=0.05):
    # Доверительный интервал для среднего при известном σ:
    z_crit = norm.ppf(1 - alpha/2)
    se = sigma / math.sqrt(n)
    lower = sample_mean_value - z_crit * se
    upper = sample_mean_value + z_crit * se
    return lower, upper

def ci_mean_unknown_sigma(sample_mean_value, sample_std_value, n, alpha=0.05):
    # Доверительный интервал для среднего при неизвестном σ:
    t_crit = t.ppf(1 - alpha/2, df=n-1)
    se = sample_std_value / math.sqrt(n)
    lower = sample_mean_value - t_crit * se
    upper = sample_mean_value + t_crit * se
    return lower, upper

def p_value_two_sided_t(t_value, n):
    # Двусторонний p-value для t-распределения:
    df = n - 1
    return 2 * (1 - t.cdf(abs(t_value), df))
