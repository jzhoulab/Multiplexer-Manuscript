# Multiplexer-Manuscript

This Repo contains the code to reproduce the results and figures found in the Multiplexer manuscript. 

The ./Supp directory contains a .ipynb notebook that reproduces the Beluga and BelugaMutliplexer predictions found in the supplementary figures.

The ./comparison directory contains the code to produce Figure c. and Supplementary Figure 2. A INFO.txt file is provided within the directory to explain how the figures and values were deteremined.

The ./experimental directory contains the code to reproduce Figure d. which comparisons Beluga and BelugaMultiplexer prediction performance on Ataq-seq and DNase-seq datasets.

The ./speedtest directory demonstrates how to create Figure b. which is a speed comparison between Beluga and BelugaMuliplexer models. 

# Data setup

To reproduce these results, first `cd` into the Multiplexer-Manuscript directory and download the data file from Zenodo with

```sh
wget https://zenodo.org/record/7500327/files/data.zip?download=1
```

Alternatively use this [link](https://zenodo.org/record/7500327#.Y7ZMX-zMKrw) to download the data off the website and manually move it into the Multiplexer-Manuscript directory. 

These files are zipped and can be unzipped with 

```sh
unzip data.zip?download=1
```
  


