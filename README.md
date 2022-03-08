# dc-DeepMSI

dc-DeepMSI provides a deep learning-based method to identify underlying metabolic heterogeneity from high-dimensional mass spectrometry imaging data. Developer is Lei Guo from Laboratory of Biomedical Network, Department of Electronic Science, Xiamen University of China.

# Overview of dc-DeepMSI

<div align=center>
<img src="https://user-images.githubusercontent.com/70273368/156913023-9654e8b0-1cb7-494f-8715-02d8d172daca.png" width="600" height="500" /><br/>
</div>
Architecture of dc-DeepMSI model. The upper half part is dimensionality reduction module which reduces a high-dimensional MSI data to a low-dimensional feature map. The dimension reduction module is implemented by an autoencoder which consists of two fully connection layers in both encoder and decoder blocks. The lower half part is feature clustering module which is consisted of two CNN networks and two ensemble CNN networks. Each CNN network consists of a feature extraction (FE) block and an argmax classification. The cluster label from one ensemble CNN network is feed into its counterpart CNN network by loss function to stabilize the segmentation result. When dc-DeepMSI reaches convergence, the four CNN networks will also converge to a similar cluster label.

# Quick start

## Input
 * The raw MSI data is collected using Bruker RapifleX MALDI Tissuetyper. SCiLS Lab vendor software is used to read and export MSI data to .imzML files. Detailed of imzML format can be found in https://ms-imaging.org/imzml/

 * Taking msi data of fetuse mouse as an example, you can download it by following scripts：
 
```
https://
```

## Run

### Step 1 Preprocessing raw data

Here, MSI data preprocessing including spectral alignment, peak detection, peak binning, peak filtering and peak pooling. Among them, spectral alignment, peak detection, peak binning are achieved using R package "MALDIquant", peak filtering and peak pooling are carried out by in-house Python scripts.

### Step 2 Run dc-DeepMSI model

