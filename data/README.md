# Data

## Local datasets path
It is important to keep data separate from the main repository. We suggest creating a dedicated path for datasets and models, along with associated files such as checkpoints and videos, to ensure organization and accessibility.
```
mkdir -p $HOME/datasets/ready
cd $HOME/datasets/ready
```

## References
* This work adopts ["Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update"](https://doi.org/10.1148/ryai.240300)
* Data types https://pytorch.org/docs/stable/tensors.html 
* PyTorch essentially defines nine CPU tensor types and nine GPU tensor types [:link:](https://stackoverflow.com/questions/60440292/runtimeerror-expected-scalar-type-long-but-found-float)
