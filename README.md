  

# Text2Mol

This is code for the paper [Text2Mol: Cross-Modal Molecule Retrieval with Natural Language Queries](https://aclanthology.org/2021.emnlp-main.47/)


![Task Example](https://github.com/cnedwards/text2mol/blob/master/misc/task2.PNG?raw=true)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/text2mol-cross-modal-molecule-retrieval-with/cross-modal-retrieval-on-chebi-20)](https://paperswithcode.com/sota/cross-modal-retrieval-on-chebi-20?p=text2mol-cross-modal-molecule-retrieval-with)

### Installation

Code is written in Python 3. Packages are shown in code/packages.txt. However, the following should suffice:
> pytorch
> pytorch-geometric
> transformers
> scikit-learn
> numpy

For processing .sdf files, we recommend [RDKit](https://www.rdkit.org/docs/GettingStartedInPython.html).

For ranker_threshold.py:
> matplotlib

### Files

| File      | Description |
| ----------- | ----------- |
| main.py      | Train Text2Mol.       |
| main_parallel.py   | A lightly-tested parallel version.        |
| ranker.py   | Rank output embeddings.        |
| ensemble.py   | Rank ensemble of output embeddings.        |
| test_example.py   | Runs a version of the model that you can query with arbitrary inputs for testing.        |
| extract_embeddings.py   | Extract embeddings or rules from a specific checkpoint.        |
| ranker_threshold.py   | Rank output embeddings and plot cosine score vs. ranking.        |
| models.py   | The three model definitions: MLP, GCN, and Attention.        |
| losses.py   | Losses used for training.        |
| dataloaders.py   | Code for loading the data.        |
| notebooks   | Jupyter Notebooks/Google Collab implementations.        |


### Example commands:

To train the model:

> python code/main.py --data data --output_path test_output --model MLP --epochs 40 --batch_size 32

ranker.py can be used to rank embedding outpoints. ensemble.py ranks the ensemble of multiple embeddings.  

> python code/ranker.py test_output/embeddings --train --val --test

> python code/ensemble.py test_output/embeddings GCN_outputs/embeddings --train --val --test

To run example queries given a model checkpoint for the MLP model:

> python code/test_example.py test_output/embeddings/ data/ test_output/CHECKPOINT.pt

To get embeddings from a specific checkpoint:

> python code/extract_embeddings.py --data data --output_path embedding_output_dir --checkpoint test_output/CHECKPOINT.pt --model MLP --batch_size 32

To plot cosine score vs ranking:

> python code/ranker_threshold.py test_output/embeddings --train --val --test --output_file threshold_image.png

All code has been rewritten as Python files so far except association_rules.ipynb.


### Data: *ChEBI-20*

Data can be found in "data/". Files directly used in the dataloaders are "training.txt", "val.txt", and "test.txt". These include the CIDs (pubchem compound IDs), mol2vec embeddings, and ChEBI descriptions. SDF (structural data file) versions are also available. 

Thanks to [PubChem](https://pubchem.ncbi.nlm.nih.gov/) and [ChEBI](https://www.ebi.ac.uk/chebi/) for freely providing access to their databases. 


### Citation
If you found our work useful, please cite:
```bibtex
@inproceedings{edwards2021text2mol,
  title={Text2Mol: Cross-Modal Molecule Retrieval with Natural Language Queries},
  author={Edwards, Carl and Zhai, ChengXiang and Ji, Heng},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages={595--607},
  year={2021},
  url = {https://aclanthology.org/2021.emnlp-main.47/}
}
```


![Poster](https://github.com/cnedwards/text2mol/blob/master/misc/Text2Mol_EMNLP_poster.png?raw=true)
