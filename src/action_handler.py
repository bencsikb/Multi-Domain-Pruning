import numpy as np
import logging

from types import SimpleNamespace
from pruning.channel_selection.alpha_functions import ActionFunc
from pruning.channel_selection.alpha_pdf import PDFGenerator


class ActionHandler:
    def __init__(self, conf: SimpleNamespace, n_prunable_layers: int) -> None:
        self._conf = conf
        self._n_prunable_layers = n_prunable_layers

        self._possible_alphas = None
        self._pdf_generator = None


    def define_alpha_list(self, to_save=False) -> None: 
        """ Get the possible action (alpha) values in a list. """

        action_generator = ActionFunc(self._conf)
        if to_save:
            action_generator.plot_and_save(self._conf.samples.save_path)

        self._possible_alphas = action_generator.alphas
    

    def define_alpha_pdfs(self, to_save=False) -> None:
        
        pdf_generator = PDFGenerator(self._n_prunable_layers, self._possible_alphas, transition_index=self._conf.alpha.pdf_trans_index, factor=self._conf.alpha.pdf_factor)
        if to_save:
            pdf_generator.plot_and_save(self._conf.samples.save_path)

        self._pdf_generator = pdf_generator


    def choose_alpha(self, layer_idx,  data=None, sample_handler=None):
        """ Choose an alpha value from the possible values list, based on the PDF defined for the given layer.
            Check the skip rules, and select alpha = 0 if any skip ruke is applied for the layer.
            If data and sample_handler are set: Check if the data sample with the selected alpha already exists, and repeat until finding a non-existinig sample. 
                                                Otherwise, just return the first sampled alpha.

        Returns:
            alpha (float): rounded to 2 decimals
            is_existing_sample (bool): True if all the possible alpha values construct an already existing sample
       
        """

        n_possible_alphas = len(self._possible_alphas)

        is_applied_skip = self._apply_skip_rules(layer_idx)

        if (data is not None) and (sample_handler is not None): 
            data_temp = data.copy()
            tried_alphas = []
            is_existing_sample = True
            while is_existing_sample:

                alpha = 0.0 if is_applied_skip else self._pdf_generator.sample_from_pdf(layer_idx)
                if is_applied_skip: logging.info(f"Skiprule applied for layer {layer_idx}.")

                if alpha not in tried_alphas:
                    tried_alphas.append(alpha)
                    data_temp.loc[layer_idx, 'alpha'] = alpha  
                    is_existing_sample = sample_handler.is_existing_sample(data_temp)
                
                if (len(tried_alphas) == n_possible_alphas) or is_applied_skip:  
                        break 
        else: 

            alpha = 0.0 if is_applied_skip else self._pdf_generator.sample_from_pdf(layer_idx)
            is_existing_sample = None
           
        logging.info(f"{alpha = }, {is_existing_sample = }")
        
        return alpha, is_existing_sample
    
    def force_zero(self, layer_idx, data, sample_handler):
        """ Force zero alpha & check if the sample exists already.
        """

        alpha = 0.0

        data_temp = data.copy()
        data_temp.loc[layer_idx, 'alpha'] = alpha  
        is_existing_sample = sample_handler.is_existing_sample(data_temp)

        return alpha, is_existing_sample
  

    def _apply_skip_rules(self, layer_idx) -> bool:

        is_applied = False
        skipunder = getattr(self._conf.channel_selection, "skipunder", None)
        skipmod = getattr(self._conf.channel_selection, "skipmod", None)

        if (skipunder is not None) and (layer_idx < skipunder):
            is_applied = True
        elif skipmod is not None:

            if not isinstance(skipmod, list):
                skipmod = [skipmod]

            for mod in skipmod:
                if layer_idx % mod == 0:
                    is_applied = True
                    break 

        return is_applied

   