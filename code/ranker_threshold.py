
#Note: There is a very slight variance in the results which arises because the molecule encoder cannot distinguish things like isotopes.

#python ranker.py ../softmax_CLIP/embeddings/GCN1/ --test

import os
import os.path as osp
import shutil
import math

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

import argparse

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Ensemble of Text2mol embeddings.')
parser.add_argument('dir', metavar='directory', type=str, nargs=1,
                    help='directory where embeddings are located')
parser.add_argument('--train', action='store_const', const=True, help="calculate training split ranks")
parser.add_argument('--val', action='store_const', const=True, help="calculate validation split ranks")
parser.add_argument('--test', action='store_const', const=True, help="calculate test split ranks")
parser.add_argument('--output_file', type=str, nargs='?', default='ranker_threshold.png',
                    help='file to save output image.')

args = parser.parse_args()
dir = args.dir[0]
output_file = args.output_file


cids_train = np.load(osp.join(dir, "cids_train.npy"), allow_pickle=True)
cids_val = np.load(osp.join(dir, "cids_val.npy"), allow_pickle=True)
cids_test = np.load(osp.join(dir, "cids_test.npy"), allow_pickle=True)

text_embeddings_train = np.load(osp.join(dir, "text_embeddings_train.npy"))
text_embeddings_val = np.load(osp.join(dir, "text_embeddings_val.npy"))
text_embeddings_test = np.load(osp.join(dir, "text_embeddings_test.npy"))

chem_embeddings_train = np.load(osp.join(dir, "chem_embeddings_train.npy"))
chem_embeddings_val = np.load(osp.join(dir, "chem_embeddings_val.npy"))
chem_embeddings_test = np.load(osp.join(dir, "chem_embeddings_test.npy"))

print('Loaded embeddings')


#combine all splits:

all_text_embbedings = np.concatenate((text_embeddings_train, text_embeddings_val, text_embeddings_test), axis = 0)
all_mol_embeddings = np.concatenate((chem_embeddings_train, chem_embeddings_val, chem_embeddings_test), axis = 0)
    
all_cids = np.concatenate((cids_train, cids_val, cids_test), axis = 0)


n_train = len(cids_train)
n_val = len(cids_val)
n_test = len(cids_test)
n = n_train + n_val + n_test

offset_val = n_train
offset_test = n_train + n_val



#I wrote a multithreaded version of the cosine similarity for something else. I can upload it if needed.

#Create efficient cosine calculator
def memory_efficient_similarity_matrix_custom(func, embedding1, embedding2, chunk_size = 1000):
    rows = embedding1.shape[0]
    
    num_chunks = int(np.ceil(rows / chunk_size))
    
    for i in range(num_chunks):
        end_chunk = (i+1)*(chunk_size) if (i+1)*(chunk_size) < rows else rows #account for smaller chunk at end...
        yield func(embedding1[i*chunk_size:end_chunk,:], embedding2)

text_chem_cos = memory_efficient_similarity_matrix_custom(cosine_similarity, text_embeddings_train, all_mol_embeddings)
text_chem_cos_val = memory_efficient_similarity_matrix_custom(cosine_similarity, text_embeddings_val, all_mol_embeddings)
text_chem_cos_test = memory_efficient_similarity_matrix_custom(cosine_similarity, text_embeddings_test, all_mol_embeddings)



#Calculate Ranks:
if args.train:
    tr_avg_ranks = np.zeros((n_train, n))
if args.val:
    val_avg_ranks = np.zeros((n_val, n))
if args.test:
    test_avg_ranks = np.zeros((n_test, n))

ranks_train = []
ranks_val = []
ranks_test = []

cosine_vs_rank = []

def get_ranks(text_chem_cos, ranks_avg, offset, split= ""):
    ranks_tmp = []
    j = 0 #keep track of all loops
    for l, emb in enumerate(text_chem_cos):
        for k in range(emb.shape[0]):
            cid_locs = np.argsort(emb[k,:])[::-1]
            ranks = np.argsort(cid_locs) 
            
            ranks_avg[j,:] = ranks_avg[j,:] + ranks 
            
            rank = ranks[j+offset] + 1
            ranks_tmp.append(rank)
            
            cosine_vs_rank.append((np.max(emb[k,:]), rank))

            j += 1
            if j % 1000 == 0: print(j, split+" processed")

    return np.array(ranks_tmp)

def print_ranks(ranks, split):

    print(split+" Model:")
    print("Mean rank:", np.mean(ranks))
    print("Hits at 1:", np.mean(ranks <= 1))
    print("Hits at 10:", np.mean(ranks <= 10))
    print("Hits at 100:", np.mean(ranks <= 100))
    print("Hits at 500:", np.mean(ranks <= 500))
    print("Hits at 1000:", np.mean(ranks <= 1000))

    print("MRR:", np.mean(1/ranks))
    print()


if args.train:
    ranks_tmp = get_ranks(text_chem_cos, tr_avg_ranks, offset=0, split="train")
    print_ranks(ranks_tmp, split="Training")
    ranks_train = ranks_tmp

if args.val:
    ranks_tmp = get_ranks(text_chem_cos_val, val_avg_ranks, offset=offset_val, split="val")
    print_ranks(ranks_tmp, split="Validation")
    ranks_val = ranks_tmp

if args.test:
    ranks_tmp = get_ranks(text_chem_cos_test, test_avg_ranks, offset=offset_test, split="test")
    print_ranks(ranks_tmp, split="Test")
    ranks_test = ranks_tmp

cosines = [cr[0] for cr in cosine_vs_rank]
ranks = [cr[1] for cr in cosine_vs_rank]

plt.scatter(cosines, ranks)
plt.xlabel('Cosine Similarity')
plt.ylabel('Ranks')
plt.savefig(output_file)
plt.show()




