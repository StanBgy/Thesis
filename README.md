# Thesis


This is the repository for my thesis project, avalable here [add link when available] 


Full scripts to install the dependencies and setup of the file systems will be provided soon. 

The data can be downloaded directly from the paper: https://www.nature.com/articles/s41593-021-00962-x , by setting up an AWS account

## Usage

The pipeline goes as follows: 

`load_betas.py` loads the data, `create_rdm.py` creates both the RDM and the MDS, `apply_rotation.py` rotates the MDSs, `fit_params_inverse.py` does the gaussian fitting, and `create_models_bestroi.py` creates the visual system models. 
Finally, `export_to_corticalsurface.ipynb` exports the data from the visual system models to neuroimaging data, that can be used in matlab to make the project. I also added the `distances_mds.py` function here to compute the distances after fitting. 

Finally, the `main.py` script runs everything in one go. 

## Notebooks 

The repository contains many notebooks that we used to run some anaylsis, sometimes before doing the voxel fitting (e.g.: the `flips.ipynb` was used to find which MDS to rotate to. It is not needed for running the rest anaylsis, but 
anyone is free to check what we did or used it on new data). The most important notebooks are `export_to_corticalsurface.ipynb`, as mentionned above, and `compute_correlations.ipynb`, which deals with the correlation between cortical surface
and MDS distances. This require the distances on the cortical surface (`meshes_and_distances.m`) and on the MDS (`distances_mds.py`) to be computed. 

### matlab

The matlab code can be found in the matlab repository. I decided to separate them due to the large amount of dependencies required to run the matlab code. The `matlab` folder in this repo contains the data used then in the
matlab analysis script, to run our ANOVA tests and create the graphs used in the manuscript. It is mainly used to make the projections on the cortical surface ( `drawrois.m`), calculating the distances between voxels on the cortical surface (`meshes_and_distances.m`) and do the ANOVA tests (` analysis.m`)

Some dependencies are needed to run this code, and will be added very soon 

