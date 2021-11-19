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


text_trunc_length = 256

mol_trunc_length = 512 #attention model only



import argparse

parser = argparse.ArgumentParser(description='Extract embeddings from a given model checkpoint.')
parser.add_argument('--data', metavar='data', type=str, 
                    help='directory where data is located')
parser.add_argument('--output_path', metavar='output_path', type=str, 
                    help='directory where data is located')
parser.add_argument('--checkpoint', type=str,
                    help='path to checkpoint file')
parser.add_argument('--model', type=str, default='MLP', nargs='?',
                    help="model type from 'MLP, 'GCN', 'Attention'")
parser.add_argument('--batch_size', type=int, default=32,
                    help='Size of data batch.')
parser.add_argument('--mol_trunc_length', type=int, nargs='?', default=512,
                    help='Molecule truncation length.')
parser.add_argument('--text_trunc_length', type=int, nargs='?', default=256,
                    help='Text truncation length.')

args = parser.parse_args()
data_path = args.data
output_path = args.output_path
CHECKPOINT = args.checkpoint
MODEL = args.model

BATCH_SIZE = args.batch_size

text_trunc_length = args.text_trunc_length
mol_trunc_length = args.mol_trunc_length #attention model only

path_token_embs = osp.join(data_path, "token_embedding_dict.npy")
path_train = osp.join(data_path, "training.txt")
path_val = osp.join(data_path, "val.txt")
path_test = osp.join(data_path, "test.txt")
path_molecules = osp.join(data_path, "ChEBI_defintions_substructure_corpus.cp")

graph_data_path = osp.join(data_path, "mol_graphs.zip")


if MODEL == "MLP":
    gd = GenerateData(text_trunc_length, path_train, path_val, path_test, path_molecules, path_token_embs)

    # Parameters
    params = {'batch_size': BATCH_SIZE,
            'num_workers': 1}

    training_generator, validation_generator, test_generator = get_dataloader(gd, params)

    model = MLPModel(ninp = 768, nhid = 600, nout = 300)

elif MODEL == "GCN":
    gd = GenerateData(text_trunc_length, path_train, path_val, path_test, path_molecules, path_token_embs)

    # Parameters
    params = {'batch_size': BATCH_SIZE,
            'num_workers': 1}

    training_generator, validation_generator, test_generator = get_dataloader(gd, params)
    
    graph_batcher_tr, graph_batcher_val, graph_batcher_test = get_graph_data(gd, graph_data_path)

    model = GCNModel(num_node_features=graph_batcher_tr.dataset.num_node_features, ninp = 768, nhid = 600, nout = 300, graph_hidden_channels = 600)

elif MODEL == "Attention":
    gd = GenerateDataAttention(text_trunc_length, path_train, path_val, path_test, path_molecules, path_token_embs)

    # Parameters
    params = {'batch_size': BATCH_SIZE,
            'num_workers': 1}

    training_generator, validation_generator, test_generator = get_attention_dataloader(gd, params)

    graph_batcher_tr, graph_batcher_val, graph_batcher_test = get_attention_graph_data(gd, graph_data_path, mol_trunc_length)

    model = AttentionModel(num_node_features=graph_batcher_tr.dataset.num_node_features, ninp = 768, nout = 300, nhead = 8, nhid = 512, nlayers = 3, 
        graph_hidden_channels = 768, mol_trunc_length=mol_trunc_length, temp=0.07)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

if MODEL == "Attention":
    tmp = model.set_device(device)
else:
    tmp = model.to(device)

model.eval()
model.load_state_dict(torch.load(CHECKPOINT))


cids_train = np.array([])
cids_val = np.array([])
cids_test = np.array([])
chem_embeddings_train = np.array([])
text_embeddings_train = np.array([])
chem_embeddings_val = np.array([])
text_embeddings_val = np.array([])
chem_embeddings_test = np.array([])
text_embeddings_test = np.array([])

if MODEL != "Attention": #Store embeddings:
    def get_emb(d, graph_batcher = None):
        with torch.no_grad():
            cid = np.array([d['cid']])
            text_mask = torch.Tensor(d['input']['text']['attention_mask']).bool().reshape(1,-1).to(device)

            text = torch.Tensor(d['input']['text']['input_ids']).long().reshape(1,-1).to(device)
            molecule = torch.Tensor(d['input']['molecule']['mol2vec']).float().reshape(1,-1).to(device)
            
            if MODEL == "MLP":
                text_emb, chem_emb = model(text, molecule, text_mask)
            elif MODEL == "GCN":
                graph_batch = graph_batcher([d['input']['molecule']['cid']]).to(device)
                graph_batch.edge_index = graph_batch.edge_index.reshape((2,-1))
                text_emb, chem_emb = model(text, graph_batch, text_mask)
            
            chem_emb = chem_emb.cpu().numpy()
            text_emb = text_emb.cpu().numpy()

        return cid, chem_emb, text_emb

    for i, d in enumerate(gd.generate_examples_train()):

        if MODEL == "MLP":
            cid, chem_emb, text_emb = get_emb(d)
        elif MODEL == "GCN":
            cid, chem_emb, text_emb = get_emb(d, graph_batcher_tr)

        cids_train = np.concatenate((cids_train, cid)) if cids_train.size else cid
        chem_embeddings_train = np.concatenate((chem_embeddings_train, chem_emb)) if chem_embeddings_train.size else chem_emb
        text_embeddings_train = np.concatenate((text_embeddings_train, text_emb)) if text_embeddings_train.size else text_emb

        if (i+1) % 1000 == 0: print(i+1, "embeddings processed")

        
    print("Training Embeddings done:", cids_train.shape, chem_embeddings_train.shape)

    for d in gd.generate_examples_val():
        if MODEL == "MLP":
            cid, chem_emb, text_emb = get_emb(d)
        elif MODEL == "GCN":
            cid, chem_emb, text_emb = get_emb(d, graph_batcher_val)

        cids_val = np.concatenate((cids_val, cid)) if cids_val.size else cid
        chem_embeddings_val = np.concatenate((chem_embeddings_val, chem_emb)) if chem_embeddings_val.size else chem_emb
        text_embeddings_val = np.concatenate((text_embeddings_val, text_emb)) if text_embeddings_val.size else text_emb

    print("Validation Embeddings done:", cids_val.shape, chem_embeddings_val.shape)

    for d in gd.generate_examples_test():

        if MODEL == "MLP":
            cid, chem_emb, text_emb = get_emb(d)
        elif MODEL == "GCN":
            cid, chem_emb, text_emb = get_emb(d, graph_batcher_test)

        cids_test = np.concatenate((cids_test, cid)) if cids_test.size else cid
        chem_embeddings_test = np.concatenate((chem_embeddings_test, chem_emb)) if chem_embeddings_test.size else chem_emb
        text_embeddings_test = np.concatenate((text_embeddings_test, text_emb)) if text_embeddings_test.size else text_emb

    print("Test Embeddings done:", cids_test.shape, chem_embeddings_test.shape)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    np.save(osp.join(output_path, "cids_train.npy"), cids_train)
    np.save(osp.join(output_path, "cids_val.npy"), cids_val)
    np.save(osp.join(output_path, "cids_test.npy"), cids_test)
    np.save(osp.join(output_path, "chem_embeddings_train.npy"), chem_embeddings_train)
    np.save(osp.join(output_path, "chem_embeddings_val.npy"), chem_embeddings_val)
    np.save(osp.join(output_path, "chem_embeddings_test.npy"), chem_embeddings_test)
    np.save(osp.join(output_path, "text_embeddings_train.npy"), text_embeddings_train)
    np.save(osp.join(output_path, "text_embeddings_val.npy"), text_embeddings_val)
    np.save(osp.join(output_path, "text_embeddings_test.npy"), text_embeddings_test)

else: #Save association rules
    #Extract attention:
    last_decoder = model.text_transformer_decoder.layers[-1]

    mha_weights = {}
    def get_activation(name):
        def hook(model, input, output):
            mha_weights[cid] = output[1].cpu().detach().numpy()
        return hook


    handle = last_decoder.multihead_attn.register_forward_hook(get_activation(''))

    #Go through data to actually get the rules
    for i,d in enumerate(gd.generate_examples_train()):

        batch = d['input']

        cid = d['cid']
        text_mask = torch.Tensor(batch['text']['attention_mask']).bool().reshape(1,-1).to(device)

        text = torch.Tensor(batch['text']['input_ids']).long().reshape(1,-1).to(device)
        graph_batch, molecule_mask = graph_batcher_tr([batch['molecule']['cid']])
        graph_batch = graph_batch.to(device)
        molecule_mask = molecule_mask.to(device)
        graph_batch.edge_index = graph_batch.edge_index.reshape((2,-1))
            
        out = model(text, graph_batch, text_mask, molecule_mask)
        
        #for memory reasons
        mol_length = graph_batch.x.shape[0]
        text_input = gd.text_tokenizer(gd.descriptions[cid], truncation=True, padding = 'max_length', 
                                            max_length=gd.text_trunc_length - 1)
        text_length = np.sum(text_input['attention_mask'])
        
        mha_weights[cid] = mha_weights[cid][0,:text_length, :mol_length]

        if (i+1) % 1000 == 0: print("Training sample", i+1, "attention extracted.")

    for i,d in enumerate(gd.generate_examples_val()):

        batch = d['input']

        cid = d['cid']
        text_mask = torch.Tensor(batch['text']['attention_mask']).bool().reshape(1,-1).to(device)

        text = torch.Tensor(batch['text']['input_ids']).long().reshape(1,-1).to(device)
        graph_batch, molecule_mask = graph_batcher_val([batch['molecule']['cid']])
        graph_batch = graph_batch.to(device)
        molecule_mask = molecule_mask.to(device)
        graph_batch.edge_index = graph_batch.edge_index.reshape((2,-1))
            
        
        out = model(text, graph_batch, text_mask, molecule_mask)

        #for memory reasons
        mol_length = graph_batch.x.shape[0]
        text_input = gd.text_tokenizer(gd.descriptions[cid], truncation=True, padding = 'max_length', 
                                            max_length=gd.text_trunc_length - 1)
        text_length = np.sum(text_input['attention_mask'])
        mha_weights[cid] = mha_weights[cid][0,:text_length, :mol_length]

    
        if (i+1) % 1000 == 0: print("Validation sample", i+1, "attention extracted.")
    
    for i,d in enumerate(gd.generate_examples_test()):

        batch = d['input']

        cid = d['cid']
        text_mask = torch.Tensor(batch['text']['attention_mask']).bool().reshape(1,-1).to(device)

        text = torch.Tensor(batch['text']['input_ids']).long().reshape(1,-1).to(device)
        graph_batch, molecule_mask = graph_batcher_test([batch['molecule']['cid']])
        graph_batch = graph_batch.to(device)
        molecule_mask = molecule_mask.to(device)
        graph_batch.edge_index = graph_batch.edge_index.reshape((2,-1))
            
        
        out = model(text, graph_batch, text_mask, molecule_mask)

        #for memory reasons
        mol_length = graph_batch.x.shape[0]
        text_input = gd.text_tokenizer(gd.descriptions[cid], truncation=True, padding = 'max_length', 
                                            max_length=gd.text_trunc_length - 1)
        text_length = np.sum(text_input['attention_mask'])
        mha_weights[cid] = mha_weights[cid][0,:text_length, :mol_length]

    
        if (i+1) % 1000 == 0: print("Test sample", i+1, "attention extracted.")


    with open(osp.join(output_path, "mha_weights.pkl"), 'wb') as fp:
        pickle.dump(mha_weights, fp)
