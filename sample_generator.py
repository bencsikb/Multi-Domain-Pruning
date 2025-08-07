import os
import logging
import argparse 

from model.yolo_handler import YoloHandler
from src.sample_handler import SampleHandler
from src.action_handler import ActionHandler
from utils.config_parser import ConfigParser
from pruning.channel_selection.channel_selector import ChannelSelector
from pruning.model_pruner.stepwise_pruner import StepWisePruner

def construct_sample_id(sample_handler) -> str:
    sample_id = (
        str(sample_handler.n_samples) + "_" +
        str(sample_handler.model_counter) + "_" +
        str(sample_handler.layer_counter)
    )
    return sample_id

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/pruning/pruning_sampling.ini')
    args = parser.parse_args()

    # Read and save config file
    conf = ConfigParser.read(args.config)
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
    model_handler = YoloHandler(conf.model)

    # Determine prunable layers
    # TODO load model and check of metrics are same as in the generated config file
    model_handler.determine_prunable_layers()
  
    channel_selector = ChannelSelector(conf.channel_selection)
    pruner = StepWisePruner(model_handler, sample_handler, conf, channel_selector)

    # Define Action handler
    action_handler = ActionHandler(conf, len(model_handler.prunable_layers))
    action_handler.define_alpha_list(to_save=True)
    action_handler.define_alpha_pdfs(to_save=True)

    while sample_handler.n_samples < conf.samples.max_samples:

        pruner.reset_model_and_state()

        for i, layer in enumerate(model_handler.prunable_layers):

            sample_handler.increment_model_counter_if_needed(i)
            sample_handler.set_layer_counter(i)    
            sample_id = construct_sample_id(sample_handler)
            
            # Logging
            logging.info(f"Sample {sample_handler.n_samples}, layer {i}, ID: {sample_id}")
            logging.info(layer)

            # Load model
            pruner.increment_layer()
            pruner.update_state()
            
            # Check if the alpha_seq exists already
            alpha, is_existing_sample = action_handler.choose_alpha(i, pruner.data, sample_handler)
            logging.info(f"{alpha = }, {is_existing_sample = }")
            
            pruner.set_alpha(alpha)  
            pruner.select_indices()  
            pruner.prune_model()     
            pruner.determine_metrics(is_existing_sample)                
            pruner.update_label()        
            
            if is_existing_sample: # Don't save if pruning is only performed to create further non-existing states
                logging.info("The state already exists in the dataset. Keeping only for later use.") 
                continue
                # load the labels and check if the saved lables are the same as metrics_after
                # assert if not
            else:
                sample_handler.add_sample(pruner.data, pruner.label)

                data_save_path = os.path.join(conf.samples.save_path, "data", sample_id + ".pkl")
                label_save_path = os.path.join(conf.samples.save_path, "label", sample_id + ".pkl")

                assert not os.path.exists(data_save_path), f"Sample {sample_handler.n_samples} already exists at {data_save_path}!"
                assert not os.path.exists(label_save_path), f"Sample {sample_handler.n_samples} already exists at {label_save_path}!"

                pruner.data.to_pickle(data_save_path)
                pruner.label.to_pickle(label_save_path)



        
