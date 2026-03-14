import os
import json
import logging
from datetime import datetime

logger = logging.getLogger("ExperimentTracker")

class ExperimentTracker:
    def __init__(self, log_dir="logs/experiments"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.experiments = []

    def log_experiment(self, name: str, params: dict, is_ic: float, is_sharpe: float, os_ic: float, os_sharpe: float):
        """
        Logs an alpha research experiment to prevent overfitting (data mining).
        """
        experiment = {
            "timestamp": datetime.now().isoformat(),
            "signal_name": name,
            "parameters": params,
            "in_sample_ic": round(is_ic, 4),
            "in_sample_sharpe": round(is_sharpe, 4),
            "out_of_sample_ic": round(os_ic, 4),
            "out_of_sample_sharpe": round(os_sharpe, 4)
        }
        self.experiments.append(experiment)
        
        # Save to JSON logs
        log_file = os.path.join(self.log_dir, "alpha_experiments.json")
        
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                try:
                    data = json.load(f)
                except:
                    data = []
        else:
            data = []
            
        data.append(experiment)
        
        with open(log_file, "w") as f:
            json.dump(data, f, indent=4)
            
        logger.info(f"Experiment logged: {name} | IS IC={is_ic:.3f} | OS IC={os_ic:.3f}")
        return experiment
