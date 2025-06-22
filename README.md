# DMGCN：Spatial domain identification method based on multi-view graph convolutional network and contrastive learning  
**The data which are testd in DMGCN will be shared through URL as follows:**  
  
**(1) Human dorsolateral pre-frontal cortex (DLPFC):**  
The 10x Visium dataset comprises 12 slices, and all slices were manually annotated. The article presents the experiment result of slice 151507, with 4,221 spots and 33,538 genes, and 151674, with 3635 spots and 33,538 genes. The data links https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expressionto-libraries-of-human-crc.  
  
**(2) Human breast cancer:**  
The 10x Visium dataset comprises 3,798 spots and 36,601 genes, with 20 regions manually annotated. The data links to https://support.10xgenomics.com//spatial-gene-expression/datasets/1.1.0/V1_Breast_Cancer_Block_A_Section_1.  
  
**(3) Mouse embryo:**  
The Stereo-seq data comprises 19,527 spots and 27,106 genes, with 12 regions manually annotated for some significant spots. The data links to https://db.cngb.org/stomics/mosta/.  
  
**(4) Mouse Brain:**  
The MERFISH data comprises six samples, with 254 genes and 5-7 regions manually annotated. The samples 1 and 3 were selected to test the performance. The data links to https://doi.brainimagelibrary.org/doi/10.35077/g.21.  
  
The DLPFC and MERFISH could be found in the "data", and HBC and MOSTA are recommended to download through the links.
  
If you are confused in the data availbility, please contact us through email and we will try to solve your problems.
  
# Installation
## Please refer to the "requirements.txt" to install the environment, and run the code in Jupyter lab.  
## We present some relative packages with specific version as follows:
  
python==3.8.0
  
scanpy==1.9.8
  
anndata==0.9.2
  
h5py==3.7.0
  
numpy==1.22.4
  
numba==0.58.1
  
pandas==1.5.3
  
scikit-learn==1.3.2
  
torch==1.11.0+cu113
  
torch-geometric==2.1.0
  
torchvision==0.12.0+cu113
  
**For the other modules of Pytorch, we recommend to install them in Python 3.8 and Linux via this link https://pytorch-geometric.com/whl/, and these modules' version as follows:**
  
torch-cluster==1.6.9
  
torch-scatter==2.0.9
  
torch-sparse==0.6.13
  
torch-spline-conv==1.2.1
