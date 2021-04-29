# Utilities-for-The-Frugal-Labeler
This repository bundles the implementations to the methods used in [Methods for the frugal labeler: Multi-class semantic segmentation on heterogeneous labels](https://osf.io/uyk79/).

## Structure
#### [Network_Training.ipynb](Network_Training.ipynb) 
Exemplary notebook which uses the tools provided in this repository.
#### [pymodules/heterogeneous_mask_iterator.py](pymodules/heterogeneous_mask_iterator.py)
Implementation of the Heteregenous Mask iterator introduced in the paper. Fully configurable and usable with TensorFlow/Keras. 
#### [pymodules/adaptive_objective_functions.py](pymodules/adaptive_objective_functions.py)
Implementation of the loss functions introduced in the paper.
#### [pymodules/unet_model.py](pymodules/unet_model.py): 
Implementation of the UNet model used in the paper.
#### [pymodules/data.py](pymodules/data.py)
Module which provides training- and test-generators ready to be used with Keras.
#### [pymodules/create_one_hot_encoded_map_from_mask.py](pymodules/create_one_hot_encoded_map_from_mask.py)
Helper module which creates one hot encoded tensors from images (used for training).

## Contact
For questions and remarks feel free to contact [Mark Schutera](https://github.com/schutera) or [Luca Rettenberger](https://github.com/lrettenberger).
