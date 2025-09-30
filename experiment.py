
import numpy as np
import sys
import logging
from typing import Tuple, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LPRobustConformalPredictor:
    """
    Implementation of Lévy-Prokhorov Robust Conformal Prediction for time series data
    Based on the paper: "Conformal Prediction under Lévy–Prokhorov Distribution Shifts"
    """
    
    def __init__(self, epsilon: float, rho: float, alpha: float = 0.1):
        """
        Initialize the LP robust conformal predictor
        
        Args:
            epsilon: Local perturbation parameter (LP parameter)
            rho: Global perturbation parameter (LP parameter)  
            alpha: Miscoverage level (default: 0.1 for 90% coverage)
        """
        self.epsilon = epsilon
        self.rho = rho
        self.alpha = alpha
        self.calibration_scores = None
        self.quantile_wc = None
        
    def compute_scores(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Compute nonconformity scores using absolute error
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            
        Returns:
            Array of nonconformity scores
        """
        try:
            scores = np.abs(predictions - targets)
            return scores
        except Exception as e:
            logger.error(f"Error computing scores: {e}")
            sys.exit(1)
    
    def fit(self, calibration_scores: np.ndarray):
        """
        Fit the conformal predictor using calibration scores
        
        Args:
            calibration_scores: Nonconformity scores from calibration data
        """
        try:
            self.calibration_scores = calibration_scores
            n = len(calibration_scores)
            
            # Compute worst-case quantile using Proposition 3.4
            beta = 1 - self.alpha
            if self.rho >= 1 - beta:
                logger.warning("rho >= 1-beta, quantile becomes trivial")
                self.quantile_wc = np.max(calibration_scores)
            else:
                # Adjust quantile level for finite sample correction
                adjusted_beta = beta + (beta - self.rho - 2) / n
                quantile_level = min(1.0, adjusted_beta + self.rho)
                
                # Compute empirical quantile
                empirical_quantile = np.quantile(calibration_scores, quantile_level)
                self.quantile_wc = empirical_quantile + self.epsilon
                
            logger.info(f"Fitted LP robust conformal predictor with worst-case quantile: {self.quantile_wc:.4f}")
            
        except Exception as e:
            logger.error(f"Error fitting conformal predictor: {e}")
            sys.exit(1)
    
    def predict(self, prediction_score: float) -> Tuple[float, float]:
        """
        Generate prediction interval for a new data point
        
        Args:
            prediction_score: Nonconformity score for the test point
            
        Returns:
            Tuple of (lower_bound, upper_bound) for the prediction interval
        """
        if self.quantile_wc is None:
            logger.error("Predictor not fitted. Call fit() first.")
            sys.exit(1)
            
        # For time series regression, we assume the prediction is the center
        # and the interval extends symmetrically based on the worst-case quantile
        center = 0.0  # Assuming predictions are centered around 0
        lower_bound = center - self.quantile_wc
        upper_bound = center + self.quantile_wc
        
        return lower_bound, upper_bound
    
    def compute_coverage(self, test_scores: np.ndarray) -> float:
        """
        Compute empirical coverage on test data
        
        Args:
            test_scores: Nonconformity scores from test data
            
        Returns:
            Empirical coverage percentage
        """
        if self.quantile_wc is None:
            logger.error("Predictor not fitted. Call fit() first.")
            sys.exit(1)
            
        coverage = np.mean(test_scores <= self.quantile_wc)
        return coverage

class TimeSeriesSimulator:
    """
    Simulate time series data with distribution shifts for testing LP robust conformal prediction
    """
    
    def __init__(self, n_samples: int = 1000, seed: int = 42):
        self.n_samples = n_samples
        self.seed = seed
        np.random.seed(seed)
    
    def generate_stationary_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate stationary time series data"""
        try:
            # AR(1) process: y_t = 0.8 * y_{t-1} + epsilon_t
            y = np.zeros(self.n_samples)
            epsilon = np.random.normal(0, 1, self.n_samples)
            
            for t in range(1, self.n_samples):
                y[t] = 0.8 * y[t-1] + epsilon[t]
            
            # Simple predictions (could be from any model)
            predictions = 0.8 * np.roll(y, 1)
            predictions[0] = 0  # Handle first element
            
            return predictions, y
            
        except Exception as e:
            logger.error(f"Error generating stationary data: {e}")
            sys.exit(1)
    
    def generate_shifted_data(self, shift_type: str = "covariate") -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate time series data with distribution shifts
        
        Args:
            shift_type: Type of distribution shift ("covariate", "label", "mixed")
            
        Returns:
            Tuple of (predictions, targets)
        """
        try:
            if shift_type == "covariate":
                # Covariate shift: change in input distribution
                y = np.zeros(self.n_samples)
                # Increased noise variance in second half
                epsilon1 = np.random.normal(0, 1, self.n_samples // 2)
                epsilon2 = np.random.normal(0, 2, self.n_samples - self.n_samples // 2)
                epsilon = np.concatenate([epsilon1, epsilon2])
                
                for t in range(1, self.n_samples):
                    y[t] = 0.8 * y[t-1] + epsilon[t]
                    
            elif shift_type == "label":
                # Label shift: change in output distribution
                y = np.zeros(self.n_samples)
                epsilon = np.random.normal(0, 1, self.n_samples)
                # Change in AR coefficient in second half
                for t in range(1, self.n_samples):
                    if t < self.n_samples // 2:
                        y[t] = 0.8 * y[t-1] + epsilon[t]
                    else:
                        y[t] = 0.5 * y[t-1] + epsilon[t]
                        
            elif shift_type == "mixed":
                # Mixed shift: both covariate and label shift
                y = np.zeros(self.n_samples)
                epsilon1 = np.random.normal(0, 1, self.n_samples // 2)
                epsilon2 = np.random.normal(0, 1.5, self.n_samples - self.n_samples // 2)
                epsilon = np.concatenate([epsilon1, epsilon2])
                
                for t in range(1, self.n_samples):
                    if t < self.n_samples // 2:
                        y[t] = 0.8 * y[t-1] + epsilon[t]
                    else:
                        y[t] = 0.6 * y[t-1] + epsilon[t]
            else:
                raise ValueError(f"Unknown shift type: {shift_type}")
            
            predictions = 0.8 * np.roll(y, 1)  # Simple model predictions
            predictions[0] = 0
            
            return predictions, y
            
        except Exception as e:
            logger.error(f"Error generating shifted data: {e}")
            sys.exit(1)

def run_experiment():
    """Main experiment to evaluate LP robust conformal prediction on time series data"""
    logger.info("Starting LP Robust Conformal Prediction Experiment for Time Series")
    
    try:
        # Initialize simulator
        simulator = TimeSeriesSimulator(n_samples=1000)
        
        # Generate data
        logger.info("Generating stationary calibration data...")
        calib_preds, calib_targets = simulator.generate_stationary_data()
        
        # Split calibration data
        split_idx = len(calib_preds) // 2
        calib_scores_cal = np.abs(calib_preds[:split_idx] - calib_targets[:split_idx])
        calib_scores_val = np.abs(calib_preds[split_idx:] - calib_targets[split_idx:])
        
        # Test different distribution shifts
        shift_types = ["covariate", "label", "mixed"]
        epsilon_values = [0.1, 0.5, 1.0]
        rho_values = [0.05, 0.1, 0.2]
        
        results = {}
        
        for shift_type in shift_types:
            logger.info(f"\nTesting {shift_type} shift...")
            results[shift_type] = {}
            
            # Generate test data with shift
            test_preds, test_targets = simulator.generate_shifted_data(shift_type)
            test_scores = np.abs(test_preds - test_targets)
            
            for epsilon in epsilon_values:
                for rho in rho_values:
                    if rho >= 0.9:  # Skip trivial cases
                        continue
                        
                    # Initialize and fit LP robust predictor
                    predictor = LPRobustConformalPredictor(epsilon=epsilon, rho=rho, alpha=0.1)
                    predictor.fit(calib_scores_cal)
                    
                    # Compute coverage
                    coverage = predictor.compute_coverage(test_scores)
                    interval_width = 2 * predictor.quantile_wc  # Symmetric interval
                    
                    results[shift_type][(epsilon, rho)] = {
                        'coverage': coverage,
                        'interval_width': interval_width,
                        'quantile_wc': predictor.quantile_wc
                    }
                    
                    logger.info(f"  epsilon={epsilon}, rho={rho}: Coverage={coverage:.3f}, Width={interval_width:.3f}")
        
        # Print final results
        print("\n" + "="*80)
        print("FINAL EXPERIMENT RESULTS")
        print("="*80)
        
        for shift_type in shift_types:
            print(f"\n{shift_type.upper()} SHIFT RESULTS:")
            print("-" * 50)
            
            best_config = None
            best_score = -np.inf
            
            for (epsilon, rho), metrics in results[shift_type].items():
                coverage = metrics['coverage']
                width = metrics['interval_width']
                
                # Score: balance between coverage (target: 0.9) and efficiency (smaller width)
                coverage_penalty = abs(coverage - 0.9)
                efficiency_score = 1.0 / width if width > 0 else 0
                score = efficiency_score - 10 * coverage_penalty
                
                if score > best_score:
                    best_score = score
                    best_config = (epsilon, rho, coverage, width)
                
                print(f"  (ε={epsilon}, ρ={rho}): Coverage={coverage:.3f}, Width={width:.3f}")
            
            if best_config:
                epsilon, rho, coverage, width = best_config
                print(f"\n  BEST CONFIG: ε={epsilon}, ρ={rho}")
                print(f"  Coverage: {coverage:.3f}, Interval Width: {width:.3f}")
        
        # Compare with standard conformal prediction
        print("\n" + "="*80)
        print("COMPARISON WITH STANDARD CONFORMAL PREDICTION")
        print("="*80)
        
        # Standard conformal prediction (epsilon=0, rho=0)
        std_predictor = LPRobustConformalPredictor(epsilon=0, rho=0, alpha=0.1)
        std_predictor.fit(calib_scores_cal)
        
        for shift_type in shift_types:
            test_preds, test_targets = simulator.generate_shifted_data(shift_type)
            test_scores = np.abs(test_preds - test_targets)
            
            std_coverage = std_predictor.compute_coverage(test_scores)
            std_width = 2 * std_predictor.quantile_wc
            
            # Find best LP robust configuration for this shift type
            best_lp_coverage = 0
            best_lp_width = np.inf
            for (epsilon, rho), metrics in results[shift_type].items():
                if metrics['coverage'] >= 0.85:  # Reasonable coverage
                    if metrics['interval_width'] < best_lp_width:
                        best_lp_coverage = metrics['coverage']
                        best_lp_width = metrics['interval_width']
            
            print(f"\n{shift_type.upper()} SHIFT:")
            print(f"  Standard CP: Coverage={std_coverage:.3f}, Width={std_width:.3f}")
            print(f"  Best LP Robust: Coverage={best_lp_coverage:.3f}, Width={best_lp_width:.3f}")
            
            if best_lp_coverage > 0:
                coverage_improvement = best_lp_coverage - std_coverage
                width_ratio = best_lp_width / std_width if std_width > 0 else np.inf
                print(f"  Improvement: Coverage +{coverage_improvement:.3f}, Width Ratio: {width_ratio:.3f}")
        
        logger.info("Experiment completed successfully")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_experiment()
