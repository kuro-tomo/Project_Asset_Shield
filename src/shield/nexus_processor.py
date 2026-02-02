import numpy as np

class NexusProcessor:
    """
    Mathematical engine to calculate macro-sensitivity (Nexus Score).
    Uses exponential decay weights for time-lag (k) influence.
    """
    def __init__(self, n_lags=5):
        self.n_lags = n_lags
        # Assign higher weights to more recent global price movements
        # Weights (omega): [0.5, 0.25, 0.125, 0.0625, 0.03125]
        self.omega = np.array([0.5**i for i in range(1, n_lags + 1)])

    def compute_score(self, macro_price_series: list):
        """
        Calculates the weighted sum of rate-of-change (Delta) for global variables.
        """
        if len(macro_price_series) < self.n_lags + 1:
            return 0.0
        
        # Calculate percentage changes (Delta)
        prices = np.array(macro_price_series)
        deltas = (prices[1:] - prices[:-1]) / prices[:-1]
        
        # Align deltas with time-lag weights (k)
        # Resulting score represents the 'Global Pressure' exerted on the ticker
        nexus_score = np.dot(deltas[-self.n_lags:], self.omega)
        return float(nexus_score)
