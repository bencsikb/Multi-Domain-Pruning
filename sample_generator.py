import os
import logging

from src.model.model_handler import ModelHandler
from src.sample_handler import SampleHandler
from utils.config_parser import ConfigParser
from pruning.channel_selection.channel_selector import ChannelSelector
from pruning.channel_selection.utils import choose_alpha
from pruning.model_pruner.stepwise_pruner import StepWisePruner
from pruning.channel_selection.alpha_functions import ActionFunc


if __name__ == "__main__":

    # Read and save config file
    conf = ConfigParser.read("config/pruning/pruning_sampling.ini")
    ConfigParser.save(conf, os.path.join(conf.samples.save_path, "settings.ini"))

    # Set up logging 
    log_file_path = os.path.join(conf.samples.save_path, "log.txt")
    logging.basicConfig(
        filename=log_file_path,
        filemode='a',  
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        force=True
        )

    # Load the samples df and get the n_samples 
    sample_handler = SampleHandler(conf)
    sample_handler.read_all_samples()
    
    # Load model
    model_handler = ModelHandler(conf.model)

    # Determine prunable layers
    # TODO load model and check of metrics are same as in the generated config file
    #metrics = []
    model_handler.flatten_conv_layers()
    model_handler.determine_prunable_layers()
    prunable_layers = model_handler.prunable_layers
  
    channel_selector = ChannelSelector(conf.channel_selection)
    pruner = StepWisePruner(model_handler, sample_handler, conf, channel_selector)

    del model_handler

    # Get alpha PDF
    alpha_pdf = ... # generate_pdf(n_prunable_layers, len(possible_alphas))
    action_generator = ActionFunc(conf)
    action_generator.plot_and_save()

    while sample_handler.n_samples < conf.samples.max_samples:

        pruner.reset_state()

        for i, layer in enumerate(prunable_layers):
            logging.info(f"Sample {sample_handler.n_samples}, layer {i}")

            # Load model
            pruner.reset_model()
            pruner.update_state()
            
            # Check if the alpha_seq exists already
            alpha, is_existing_sample = choose_alpha(pruner.data, i, action_generator, conf, sample_handler)    # TODO remove sample dependency

            pruner.set_alpha(alpha)  
            pruner.select_indices()       
            if not is_existing_sample:
                pruner.prune_model()
                pruner.eval_pruned_model()
            pruner.update_label(is_existing_sample)            
            
            if is_existing_sample: # Don't save if pruning is only performed to create further non-existing states
                logging.info("The state already exists in the dataset. Keeping only for later use.") 
                continue
                # load the labels and check if the saved lables are the same as metrics_after
                # assert if not
            else:
                data_save_path = os.path.join(conf.samples.save_path, "data", str(sample_handler.n_samples) + ".pkl")
                label_save_path = os.path.join(conf.samples.save_path, "label", str(sample_handler.n_samples) + ".pkl")

                assert not os.path.exists(data_save_path), f"Sample {sample_handler.n_samples} already exists at {data_save_path}!"
                assert not os.path.exists(label_save_path), f"Sample {sample_handler.n_samples} already exists at {label_save_path}!"

                pruner.data.to_pickle(data_save_path)
                pruner.label.to_pickle(label_save_path)
                sample_handler.add_sample(pruner.data, pruner.label)



        
