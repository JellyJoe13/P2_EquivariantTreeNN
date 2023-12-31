# P2 - Equivariant Tree Neural Network (ETNN)
## Project description
The aim of this project is to implement a neural network that is equivariant(later focused on invariance instead of 
equivariance) on permutation trees (generalization of PQ- and PC-trees). The implementation is based on the paper
DeepSets and ChiENN, see the report of this project for more information.

## Documentation
Documentation is available via https://jellyjoe13.github.io/P2_EquivariantTreeNN - genereated using library Sphinx.

## Report
Coming soon.

## Table of project content
This section will contain a short explanation on the top level contents of this project folder - subfolders may have
their separate ReadMe or be covered in the documentation generated by Sphinx which is linked in another section.

- **datasets**: Folder containing the base dataset and will house the generated datasets. These are not stored in GitHub
    but generated as needed using all CPU cores that are available (multiprocess library)
- **docs-config**: Housing configuration of the documentation
- **docs**: Folder housing the documentation of this projects functions, classes, etc.
- **env_setup**: Folder containing scripts to install the conda environment required to run the code of this project.
- **etnn**: Main folder in python package style that contains almost all helper functions, model definitions, etc. 
    Visit documentation to learn more.
- **notebooks**: Folder containing the notebooks in which development/experiments occurred. Read report to find out
    which notebooks correspond to which experiments
- **results**: Folder containing stored results of the main experiments will the full dataset. Contains configuration
    index and per experiment the configuration, tracked scores and for some the saved model parameters.