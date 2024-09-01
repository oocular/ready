# Data

## Local datasets path
cd $HOME/datasets/openEDS

## References
* This work adopts ["Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update"](https://doi.org/10.1148/ryai.240300)
* https://pytorch.org/docs/stable/tensors.html 
* Data types
PyTorch essentially defines nine CPU tensor types and nine GPU tensor types:
╔══════════════════════════╦═══════════════════════════════╦════════════════════╦═════════════════════════╗
║        Data type         ║             dtype             ║     CPU tensor     ║       GPU tensor        ║
╠══════════════════════════╬═══════════════════════════════╬════════════════════╬═════════════════════════╣
║ 32-bit floating point    ║ torch.float32 or torch.float  ║ torch.FloatTensor  ║ torch.cuda.FloatTensor  ║
║ 64-bit floating point    ║ torch.float64 or torch.double ║ torch.DoubleTensor ║ torch.cuda.DoubleTensor ║
║ 16-bit floating point    ║ torch.float16 or torch.half   ║ torch.HalfTensor   ║ torch.cuda.HalfTensor   ║
║ 8-bit integer (unsigned) ║ torch.uint8                   ║ torch.ByteTensor   ║ torch.cuda.ByteTensor   ║
║ 8-bit integer (signed)   ║ torch.int8                    ║ torch.CharTensor   ║ torch.cuda.CharTensor   ║
║ 16-bit integer (signed)  ║ torch.int16 or torch.short    ║ torch.ShortTensor  ║ torch.cuda.ShortTensor  ║
║ 32-bit integer (signed)  ║ torch.int32 or torch.int      ║ torch.IntTensor    ║ torch.cuda.IntTensor    ║
║ 64-bit integer (signed)  ║ torch.int64 or torch.long     ║ torch.LongTensor   ║ torch.cuda.LongTensor   ║
║ Boolean                  ║ torch.bool                    ║ torch.BoolTensor   ║ torch.cuda.BoolTensor   ║
╚══════════════════════════╩═══════════════════════════════╩════════════════════╩═════════════════════════╝

https://pytorch.org/docs/stable/tensors.html  


