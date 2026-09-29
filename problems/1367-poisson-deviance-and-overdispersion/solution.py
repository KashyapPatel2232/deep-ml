import numpy as np


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Poisson deviance, using the convention 0 * log(0) = 0."""
    # Your code here
    ratio = np.where(y > 0, y / mu, 1.0)
    log_ratio = np.log(ratio)
    
    deviance_terms = y * log_ratio - (y - mu)
    
    return float(2 * np.sum(deviance_terms))


def dispersion_ratio(y: np.ndarray, mu: np.ndarray, n_params: int) -> float:
    """Pearson chi-square divided by (n - n_params)."""
    # Your code here
    n = y.size
    pearson_chi2 = np.sum((y - mu)**2 / mu)
    return float(pearson_chi2 / (n - n_params))
