import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import logging


class ActionFunc:
    def __init__(self, conf) -> None:
        # Ensure that `scale` and `min_max_steps` are defined in `conf.alpha`
        assert hasattr(conf.alpha, "scale"), "Error: 'scale' is not defined in conf.alpha"
        assert hasattr(conf.alpha, "min_max_steps"), "Error: 'min_max_steps' is not defined in conf.alpha"
        
        scale = conf.alpha.scale
        min_max_steps = conf.alpha.min_max_steps
        self._min, self._max, self._n_steps = min_max_steps  # Unpack min, max, n_steps
        self._conf = conf

        # Choose the action function based on `scale`
        if getattr(conf.alpha, 'value_list', None) is not None:
            self.possible_alphas = conf.alpha.value_list
            self.action_func = None  # No need to generate if `value_list` is provided
        elif scale == "lin":
            self.action_func = LinspaceAction()
        elif scale == "half_sigmoid":
            self.action_func = HalfSigmoidAction()
        elif scale == "loge":
            self.action_func = LogEAction()
        else:
            print(f"The selected action function '{scale}' is not implemented. Please select from [lin, half_sigmoid].")
            sys.exit(1)
        
        self._alphas = self._generate()

    def _generate(self) -> list:
        """Delegates to the appropriate action function's generate method."""
        if self.action_func:
            return self.action_func.generate(self._min, self._max, self._n_steps)
        else:
            return self.possible_alphas  
    
    
    def plot_and_save(self, filename="action_func.png"):
        """ Plots the alpha values and saves the plot to a file.
            If the plot already exists with the given name, adds an int to the filename.            
        """

        values = self._alphas
        plt.figure(figsize=(8, 5))
        plt.plot(values, marker="o", linestyle="-")
        plt.title("Generated Sequence Plot")
        plt.xlabel("Step Index")
        plt.ylabel("Value")
        plt.grid(True)

        # Check if the file already exists and modify the filename if necessary
        base_path = self._conf.samples.save_path
        base_filename, file_extension = os.path.splitext(filename)
        counter = 1
        new_filename = filename

        while os.path.exists(os.path.join(base_path, new_filename)):
            new_filename = f"{base_filename}_{counter}{file_extension}"
            counter += 1

        # Save the plot
        plt.savefig(os.path.join(base_path, new_filename))
        plt.close()
        logging.info(f"Plot saved as {new_filename}")
    
    @property
    def alphas(self):
        return self._alphas


class LinspaceAction:
    def generate(self, min, max, n_steps) -> list:
        """Generate a linearly spaced sequence from min to max with n_steps."""
        linear_values = np.linspace(min, max, n_steps).tolist()
        return [round(val, 1) for val in linear_values]
    

class HalfSigmoidAction:
    def generate(self, min, max, n_steps) -> list:
        """Generate a half-sigmoid sequence from min to max with n_steps."""
        x_values = np.linspace(-3, 3, n_steps)  
        sigmoid_values = [1 / (1 + np.exp(-x)) for x in x_values]
        
        scaled_values = [min + (max - min) * val for val in sigmoid_values]
        
        scaled_values[0] = min
        scaled_values[-1] = max

        return [round(val, 1) for val in scaled_values]

class LogEAction:
    def generate(self, min, max, n_steps) -> list:
        """Generate a log-e sequence from min to max with n_steps, including exact endpoints."""

        x_values = np.linspace(1, np.e, n_steps)
        log_values = np.log(x_values)  # Natural log of each value in x_values
        
        min_log, max_log = log_values.min(), log_values.max()
        scaled_values = [min + (max - min) * (val - min_log) / (max_log - min_log) for val in log_values]

        return [round(val, 1) for val in scaled_values]

