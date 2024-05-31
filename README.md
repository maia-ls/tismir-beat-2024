# tismir-beat-2024
Code for research paper "Selective Annotation of Few Data for Beat Tracking of Latin American Music Using Rhythmic Features" published in TISMIR in 2024.

This repository includes functions for extracting rhythmic representations (scale transform magnitudes and onset patterns) and selective sampling schemes (vote-k, mfl, diversity, k-medoids). All functions are written in Python, with package requirements expressed in the [requirements](https://github.com/maia-ls/tismir-beat-2024/blob/main/requirements.txt) file. The only exception is the medoid-based feature selection, which is written for Matlab, but can be called through Python in a subprocess (provided that you have Matlab installed). Examples of function calling are provided in the [example notebook](https://github.com/maia-ls/tismir-beat-2024/blob/main/example.ipynb).

Some interesting references that were left out of the article for space purposes can be seen in the [appendix](https://github.com/maia-ls/tismir-beat-2024/blob/main/appendix.md).

## Citation
If you use this in your work, please consider citing:

```
@article{selective_beat,
   author = "Maia, Lucas S. and Rocamora, Mart{\'{i}}n and Biscainho, Luiz W. P. and Fuentes, Magdalena",
   title = "Selective Annotation of Few Data for Beat Tracking of Latin American Music Using Rhythmic Features",
   journal = "Transactions of the International Society for Music Information Retrieval",
   volume = "7",
   number = "1",
   month = mar,
   year = "2014",
   doi = "10.5334/tismir.170",
}
```
