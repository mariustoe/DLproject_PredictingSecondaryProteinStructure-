Target Embedding Autoencoder

This repository contains two notebooks to predict secondary protein structures - one notebook computing the main results of the Target Embedding Autoencoder and a notebook containing a standard CNN for comparison to the TEA.

To run the notebook, all you need to do is download the repo, unzip the datafiles, and change the path to your files in the top of each notebook. On a normal functioning computer with standard specs it should take appx 2-3 hours to run the models on the subset of the data.

The available data in the repository is a subset of the full dataset. The full dataset can be found at NetSurfP's website [https://services.healthtech.dtu.dk/service.php?NetSurfP-2.0]. Data from NetSurfP has to be converted to .pickle objects and only the amino acid sequences and secondary structures should be included.
