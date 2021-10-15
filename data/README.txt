The data is contained in 6 files:

(1,2,3) The mol2vec_ChEBI_20_X.txt files have lines in the following form:
CID	mol2vec embedding	Description


(4) mol_graphs.zip contain {cid}.graph files. These are formatted first with the edgelist of the graph and then substructure tokens for each node.
For example,
edgelist:
0 1
1 0
1 2
2 1
1 3
3 1

idx to identifier:
0 3537119515
1 2059730245
2 3537119515
3 1248171218


(5) ChEBI_defintions_substructure_corpus.cp contains the molecule token "sentences". It is formatted:
cid: tokenid1 tokenid2 tokenid3 ... tokenidn


(6) token_embedding_dict.npy is a dictionary mapping molecule tokens to their embeddings. It can be loaded with the following code:
import numpy as np
token_embedding_dict = np.load("token_embedding_dict.npy", allow_pickle=True)[()]