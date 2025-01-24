# ASTIFNet

The repo is the official implementation for the paper: "Lightweight adaptive spatio-temporal information fusion network for multivariate medical time series classification".


# Environment 

- `torch==2.1.2`
- `torchvision==0.16.2`
- `einops==0.8.0`
- `numpy==1.26.3`
- `opencv-python==4.10.0.84`
- `pandas==2.2.3`
- `scikit-learn==1.5.2`
- `torchsummary==1.5.1`


# Datasets
The dataset required for ASTIFNet can be obtained from [Medformer](https://github.com/DL4mHealth/Medformer).


# Experiments
To replicate our results on the APAVA, TDBRAIN, ADFTD, PTB and PTBXL datasets, run
```
bash ./scripts/run_main.sh
```

## Acknowledgement
We appreciate the following GitHub repositories for providing valuable code bases and datasets:

https://github.com/DL4mHealth/Medformer

https://github.com/thuml/Time-Series-Library

Thanks a lot for their amazing work on implementing state-of-arts time series methods!