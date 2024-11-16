
import numpy as np
import os
import matplotlib.pyplot as plt


class PDFGenerator:
    def __init__(self, n, values, transition_index, factor) -> None:
        self._n = n
        self._values = values
        self._transition_index = transition_index
        self._factor = factor

        self._pdf_list = self._generate_pdf()

    def _generate_pdf(self):
        """
        Generate a PDF for values values over n points, with dominant P(0) before transition_index.

        Returns:
        pdf_list: numpy.ndarray
            A 2D array representing the PDF over `n` points.
        """
        pdf_list = []
        for i in range(self._n):

            if i < self._transition_index:
                probs = [0.95] + [0.05 / (len(self._values) - 1)] * (len(self._values) - 1)
            else:
                adjusted_scale = (i - self._transition_index) / (self._n - self._transition_index)
                probs = [(1 - adjusted_scale) ** self._factor] + [
                    (adjusted_scale ** (v - 1)) for v in range(2, len(self._values) + 1)
                ]

            probs = np.array(probs)
            probs /= probs.sum()

            pdf_list.append(probs)

        return np.array(pdf_list)
    
    def sample_from_pdf(self, i):
        """Sample based on the precomputed probabilities in the pdf. """

        return  np.random.choice(self._values, p=self._pdf_list[i])
    

    def plot_and_save(self, path):
            # Base filename and extension
            base_filename = "alpha_pdf.png"
            file_path = os.path.join(path, base_filename)

            # Check if the file already exists, and if so, add a number to the filename
            if os.path.exists(file_path):
                base_name, ext = os.path.splitext(base_filename)
                counter = 1
                while os.path.exists(file_path):
                    file_path = os.path.join(path, f"{base_name}_{counter}{ext}")
                    counter += 1

            # Plot the PDF
            plt.figure(figsize=(10, 6))
            for v in range(len(self._values)):
                plt.plot(self._pdf_list[:, v], label=f'P({self._values[v]})')

            plt.title("Probability Distribution Function (PDF)")
            plt.xlabel("Index")
            plt.ylabel("Probability")
            plt.grid(True)
            plt.legend()

            # Save the plot
            plt.savefig(file_path)
            plt.close()
            print(f"Alpha PDF plot saved to {file_path}")