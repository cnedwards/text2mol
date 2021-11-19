import os
import os.path as osp
import shutil
import time
import csv
import math
import pickle

import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

import torch.optim as optim
from transformers.optimization import get_linear_schedule_with_warmup


from losses import contrastive_loss, negative_sampling_contrastive_loss
from models import MLPModel, GCNModel, AttentionModel
from dataloaders import get_dataloader, GenerateData, get_graph_data, get_attention_graph_data, GenerateDataAttention, get_attention_dataloader

from sklearn.metrics.pairwise import cosine_similarity



import argparse

parser = argparse.ArgumentParser(description='Query model for closest molecules.')
parser.add_argument('emb_dir', metavar='emb_dir', type=str, nargs=1,
                    help='directory where embeddings are located')
parser.add_argument('data_path', metavar='data_path', type=str, nargs=1,
                    help='directory where data is located')
parser.add_argument('checkpoint', type=str,
                    help='path to checkpoint file')
parser.add_argument('model', type=str, default='MLP', nargs='?',
                    help="model type from 'MLP, 'GCN', 'Attention'. Only MLP is known to work.")

args = parser.parse_args()
emb_dir = args.emb_dir[0]
data_path = args.data_path[0]
CHECKPOINT = args.checkpoint
MODEL = args.model

#data_path = "../name_data"
path_token_embs = None #osp.join(data_path, "token_embedding_dict.npy")
path_train = osp.join(data_path, "mol2vec_ChEBI_20_training.txt")
path_val = osp.join(data_path, "mol2vec_ChEBI_20_val.txt")
path_test = osp.join(data_path, "mol2vec_ChEBI_20_test.txt")
path_molecules = None #osp.join(data_path, "ChEBI_defintions_substructure_corpus.cp")

graph_data_path = osp.join(data_path, "mol_graphs.zip")


BATCH_SIZE = 1

text_trunc_length = 256

mol_trunc_length = 512 #attention model only

#emb_dir = "../embeddings/"
#CHECKPOINT = "../final_weights.720.pt"

if MODEL == "MLP":
    gd = GenerateData(text_trunc_length, path_train, path_val, path_test, path_molecules, path_token_embs)

    # Parameters
    params = {'batch_size': BATCH_SIZE,
            'num_workers': 1}

    training_generator, validation_generator, test_generator = get_dataloader(gd, params)

    model = MLPModel(ninp = 768, nhid = 600, nout = 300)


if torch.cuda.is_available():
    model.load_state_dict(torch.load(CHECKPOINT))
else:
    model.load_state_dict(torch.load(CHECKPOINT, map_location=torch.device('cpu')))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

if MODEL == "Attention":
    tmp = model.set_device(device)
else:
    tmp = model.to(device)



cids_train = np.load(osp.join(emb_dir, "cids_train.npy"), allow_pickle=True)
cids_val = np.load(osp.join(emb_dir, "cids_val.npy"), allow_pickle=True)
cids_test = np.load(osp.join(emb_dir, "cids_test.npy"), allow_pickle=True)

text_embeddings_train = np.load(osp.join(emb_dir, "text_embeddings_train.npy"))
text_embeddings_val = np.load(osp.join(emb_dir, "text_embeddings_val.npy"))
text_embeddings_test = np.load(osp.join(emb_dir, "text_embeddings_test.npy"))

chem_embeddings_train = np.load(osp.join(emb_dir, "chem_embeddings_train.npy"))
chem_embeddings_val = np.load(osp.join(emb_dir, "chem_embeddings_val.npy"))
chem_embeddings_test = np.load(osp.join(emb_dir, "chem_embeddings_test.npy"))

all_text_embbedings = np.concatenate((text_embeddings_train, text_embeddings_val, text_embeddings_test), axis = 0)
all_mol_embeddings = np.concatenate((chem_embeddings_train, chem_embeddings_val, chem_embeddings_test), axis = 0)
    
all_cids = np.concatenate((cids_train, cids_val, cids_test), axis = 0)


def name_to_input(name, data_generator):
    """Yields examples."""

    text_input = data_generator.text_tokenizer(name, truncation=True, padding = 'max_length', 
                                     max_length=text_trunc_length, return_tensors = 'pt')

    return {
        'cid': '',
        'input': {
            'text': {
              'input_ids': text_input['input_ids'],
              'attention_mask': text_input['attention_mask'],
            },
            'molecule' : {
                'mol2vec' : torch.zeros((1,300)),
                'cid' : ''
            }
        },
    }

model.eval()
with torch.set_grad_enabled(False):
    name = ""

    name = input("Enter description or 'stop': ")
    while name != "stop":
        
        inputs = name_to_input(name, gd)['input']
        
        text_mask = inputs['text']['attention_mask'].bool()

        text = inputs['text']['input_ids'].to(device)
        text_mask = text_mask.to(device)
        molecule = inputs['molecule']['mol2vec'].float().to(device)

        if MODEL == "MLP":
            text_out, chem_out = model(text, molecule, text_mask)
                    
        elif MODEL == "GCN":
            graph_batch = graph_batcher_val(d[0]['molecule']['cid']).to(device)
            text_out, chem_out = model(text, graph_batch, text_mask)
        

        elif MODEL == "Attention":
            graph_batch, molecule_mask = graph_batcher_val(d[0]['molecule']['cid'])
            graph_batch = graph_batch.to(device)
            molecule_mask = molecule_mask.to(device)
            labels = labels.float().to(device)
            text_out, chem_out = model(text, graph_batch, text_mask, molecule_mask)


        name_emb = text_out.cpu().numpy()
        
        sims = cosine_similarity(name_emb, all_mol_embeddings)

        cid_locs = np.argsort(sims).squeeze()[::-1]
        ranks = np.argsort(cid_locs)
        
        print(ranks[:20])
        sorted = np.argsort(ranks)
        print(all_cids[sorted[:20]])
        print([gd.descriptions[cid] for cid in all_cids[sorted[:20]]])

        print()
        print()
        name = input("Enter description or 'stop': ")






