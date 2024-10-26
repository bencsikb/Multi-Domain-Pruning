import numpy as np

def choose_alpha(data, i, alpha_pdf, conf, sample_handler):
    
    def _apply_skip_rules(conf):

        is_applied = False
        skipunder = getattr(conf.channel_selection, "skipunder", None)
        skipmod = getattr(conf.channel_selection, "skipmod", None)

        if (skipunder is not None) and (i < skipunder):
            is_applied = True
        elif (skipmod is not None) and (i % skipmod):
            is_applied = True

        return is_applied

    #TODO This sould actually go to the PDF generator
    if getattr(conf.alpha, 'value_list', None) is not None:
        possible_alphas = conf.alpha.value_list
    else:
        assert hasattr(conf.alpha, 'min_max_step'), "Alpha value list OR min, max, step values must be provided!"
        possible_alphas = np.arange(conf.alpha.min_max_step[0], conf.alpha.min_max_step[1], conf.alpha.min_max_step[2])
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


