# dc-DeepMSI

dc-DeepMSI provides a deep learning-based method to identify underlying metabolic heterogeneity from high-dimensional mass spectrometry imaging data. Developer is Lei Guo from Laboratory of Biomedical Network, Department of Electronic Science, Xiamen University of China.

# Overview of dc-DeepMSI

<div align=center>
<img src="https://user-images.githubusercontent.com/70273368/156913023-9654e8b0-1cb7-494f-8715-02d8d172daca.png" width="600" height="500" /><br/>
</div>
Architecture of dc-DeepMSI model. The upper half part is dimensionality reduction module which reduces a high-dimensional MSI data to a low-dimensional feature map. The dimension reduction module is implemented by an autoencoder which consists of two fully connection layers in both encoder and decoder blocks. The lower half part is feature clustering module which is consisted of two CNN networks and two ensemble CNN networks. Each CNN network consists of a feature extraction (FE) block and an argmax classification. The cluster label from one ensemble CNN network is feed into its counterpart CNN network by loss function to stabilize the segmentation result. When dc-DeepMSI reaches convergence, the four CNN networks will also converge to a similar cluster label.

# Quick start

## Input

 * The input is the preprocessed MSI data with two-dimensional shape [X*Y,P], where X and Y represent the pixel numbers of horizontal and vertical coordinates of MSI data, and P represents the number of ions. Taking msi data of fetuse mouse as an example, you can download it by following scripts：
 
```
https://drive.google.com/drive/folders/1ksIHUE8r8ADS90pOErroW8_XEF-sOOdP?usp=sharing
```

## Run dc-DeepMSI model

If you want to perfrom dc-DeepMSI model with "SPAT-spec" mode, taking fetus mouse data as an example, run:

```
python run.py -input_file .../data/fetus_mouse.txt --input_shape 202 107 1237 --mode SPAT-spec --output_file output
```

If you want to perfrom dc-DeepMSI model with "spat-SPEC" mode, taking fetus mouse data as an example, run:

```
python run.py -input_file .../data/fetus_mouse.txt --input_shape 202 107 1237 --mode spat-SPEC --output_file output
```

# Contact

Please contact me if you have any help: gankLei@stu.xmu.edu.cn
