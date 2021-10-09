import os
import pathlib


class Config:
    def __init__(self):

        self.device = None
        self.dataset = "HiC_Rao_10kb"
        self.model = "GraphTransformer"
        self.cell = "GM"
        self.num_chr = 23
        self.genome_len = 288091
        self.cp_resolution = 100
        self.num_nodes = 100
        self.resolution = 10000
        self.max_norm = 10
        self.lstm_nontrain = False

        self.gpu = {
            "use": False,
            "id": 0
        }

        ##########################################
        ############ Model Parameters ############
        ##########################################

        

        ##########################################
        ############ Input Directories ###########
        ##########################################

        self.hic_path = '/data2/hic_lstm/data/'
        self.sizes_file = 'chr_cum_sizes2.npy'
        self.start_end_file = 'starts.npy'

        ##########################################
        ############ Output Locations ############
        ##########################################

        self.model_dir = '../saved_models/'
        self.output_directory = '../outputs/'
        self.plot_dir = self.output_directory + 'data_plots/'
        self.processed_data_dir = self.output_directory + 'processed_data/'

        for file_path in [self.model_dir, self.output_directory, self.plot_dir, self.processed_data_dir]:
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                print("Creating directory %s" % file_path)
                pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)
            else:
                print("Directory %s exists" % file_path)
