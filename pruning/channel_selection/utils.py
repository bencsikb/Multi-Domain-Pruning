import numpy as np
from pruning.channel_selection.alpha_functions import ActionFunc

def choose_alpha(data, i, alpha_pdf, conf, sample_handler):
    
    def _apply_skip_rules(conf):

        is_applied = False
        skipunder = getattr(conf.channel_selection, "skipunder", None)
        skipmod = getattr(conf.channel_selection, "skipmod", None)

        if (skipunder is not None) and (i < skipunder):
            is_applied = True
        elif skipmod is not None:

            if not isinstance(skipmod, list):
                skipmod = [skipmod]

            for mod in skipmod:
                if i % mod == 0:
                    is_applied = True
                    break 

        return is_applied

    #TODO This sould actually go to the PDF generator
    action_generator = ActionFunc(conf)
    possible_alphas = action_generator.generate()
    action_generator.plot_and_save()
    n_possible_alphas = len(possible_alphas)

    data_temp = data.copy()
    tried_alphas = []
    is_existing_sample = True
    while is_existing_sample:

        is_applied_skip = _apply_skip_rules(conf)
        if is_applied_skip:
            alpha = 0.0
        else:
            alpha = np.random.rand() #TODO pdf
        
        if alpha not in tried_alphas:
            tried_alphas.append(alpha)
            data_temp.loc[i, 'alpha'] = alpha  
            is_existing_sample = sample_handler.is_existing_sample(data_temp)
        
        if (len(tried_alphas) == n_possible_alphas) or is_applied_skip:  
                break              

    return alpha, is_existing_sample


