  

# Text2Mol

This is code for the paper [Text2Mol: Cross-Modal Molecule Retrieval with Natural Language Queries](https://aclanthology.org/2021.emnlp-main.47/)


To run a model, use main.py. The parameters can be adjusted in this file. I will make these command line options.

Example commands:

> python code/main.py

ensemble.py / ranker.py can be used to rank embeddings.
  

> python code/ranker.py MLP_outputs/embeddings --train --val --test

> python code/ensemble.py MLP_outputs/embeddings GCN_outputs/embeddings --train --val --test

To run example queries given a model checkpoint for the MLP model:
> python code/test_example.py MLP_outputs/embeddings/ data/ MLP_outputs/final_weights.1.pt

All code has been rewritten as Python files except association_rules.ipynb

### Citation
If you found our work useful, please cite:
@inproceedings{edwards2021text2mol,
  title={Text2Mol: Cross-Modal Molecule Retrieval with Natural Language Queries},
  author={Edwards, Carl and Zhai, ChengXiang and Ji, Heng},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages={595--607},
  year={2021},
  url = {https://aclanthology.org/2021.emnlp-main.47/}
}


#### I'm still working on improving the repository.

