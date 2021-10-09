import torch
import pickle
import torch.utils.data
import time
import os
import pandas as pd
import csv
import dgl
from scipy import sparse as sp
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import networkx as nx
import hashlib
from train.config import Config


class MoleculeDGL(torch.utils.data.Dataset):
    def __init__(self, cfg, chr, split):
        self.cfg = cfg
        self.data_dir = cfg.data_dir
        self.split = split
        self.cell = cfg.cell
        self.chr = chr
        self.contact_data = None

        self.load_hic()
        self.create_hic_graphs()

        self.graph_lists = []
        self.graph_labels = []

    def create_hic_graphs(self):
        pass


class HiCDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, cfg, chr):
        t0 = time.time()

        self.cfg = cfg
        self.chr = chr
        self.num_atom_type = self.cfg.genome_len
        self.num_bond_type = self.cfg.cp_resolution

        if self.cfg.dataset == 'HiC_Rao_10kb':
            self.train = MoleculeDGL(self.cfg, self.chr, 'train')


class HiCDataset(torch.utils.data.Dataset):

    def __init__(self, name, cfg, chr):
        """
            Loading HiC dataset
        """
        start = time.time()
        print("Loading Chromosome %s from dataset %s..." % (str(chr), name))
        self.chr = chr
        self.name = name
        self.cell = cfg.cell
        self.cfg = cfg
        # self.dataset = self.get_data()

        data_dir = 'data/HiC/'

        with open(data_dir + name + '.pkl', "rb") as f:
            f = pickle.load(f)
            self.train = f[0]
            self.val = f[1]
            self.test = f[2]
            self.num_atom_type = f[3]
            self.num_bond_type = f[4]
        print('train, test, val sizes :', len(self.train), len(self.test), len(self.val))

        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(labels, dtype=torch.float64)
        labels = torch.tensor(np.array(labels)).unsqueeze(1)
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels

if __name__ == '__main__':
    cfg = Config()

    chr = 21
    HiC_data_ob = HiCDatasetDGL(cfg, chr)
