*** Tweaked for compatibility ***
# DeepFilter
For a more theoretical information please visit our paper in the journal **Biomedical Signal and Control**: 
[https://www.sciencedirect.com/science/article/abs/pii/S1746809421005899](https://www.sciencedirect.com/science/article/abs/pii/S1746809421005899 "paper")  

This repository contains the codes for DeepFilter a deep learning based Base line wander removal tool. 
This repo follows the last version of the paper where some changes on the experiment scheme were requested
by the reviewers.

Since for us reproducibility is **KEY** on research, and we understand that some folks will find and read only the 
Arxiv version, we have decided to create a separate repository 
[https://github.com/fperdigon/DeepFilter_as_in_Arxiv](https://github.com/fperdigon/DeepFilter_as_in_Arxiv "repo")
that follows the experiment scheme described in the preprint Arxiv paper 
[https://arxiv.org/pdf/2101.03423.pdf](https://arxiv.org/pdf/2101.03423.pdf "paper") 

This repository also contains other classical and deeplearning filters solutions implemented for comparison purposes.

The deep learning models were implemented using Keras/Tensorflow framework.

- [Introduction](#introduction)
- [Results](#results)
- [Installation](#installation)
- [References](#references)
- [How to cite DeepFilter](#citing-deepfilter)
- [License](#license)

# Introduction

According to the World Health Organization, around 36% of the annual deaths are associated with cardiovascular 
diseases and 90% of heart attacks are preventable. Electrocardiogram signal analysis in ambulatory electrocardiography, 
during an exercise stress test, and in resting conditions allows cardiovascular disease diagnosis. 
However, during the acquisition, there is a variety of noises that may damage the signal quality thereby compromising 
their diagnostic potential. The baseline wander is one of the most undesirable noises.
 
In this work, we propose a novel algorithm for BLW noise filtering using deep learning techniques. The model performance 
was validated using the QT Database and the MIT-BIH Noise Stress Test Database from Physionet. We implement an Inception 
inspired multibranch model that by laveraging the use og multi path modules and dilated convolutions is capable of 
filtering BLW while preserving ECG signal morphology and been computational efficient.  

The following figure shows the multipath module using dilated convolutions. 
![Multipath module](ReadmeImg/fig_filter_layer.png "Multipath module")

The following figure shows the overall model architecture.
![Model Architecture](ReadmeImg/fig_prop_arch.png "Model Architecture")

In addition, we compared our approach against state-of-the-art methods using traditional filtering procedures as well as deep learning techniques.
This other methods were implemented by us in python 
* [FIR Filter](https://github.com/fperdigon/DeepFilter/blob/master/digitalFilters/dfilters.py#L17) (using Scipy python library). Reference paper: [Francisco Perdigón Romero, Liset Vázquez Romaguera, Carlos Román Vázquez-Seisdedos, Marly Guimarães Fer-nandes Costa, João Evangelista Neto, et al.  Baseline wander removal methods for ecg signals: A comparativestudy.arXiv preprint arXiv:1807.11359, 2018.](https://arxiv.org/pdf/1807.11359.pdf)
* [IIR Filter](https://github.com/fperdigon/DeepFilter/blob/master/digitalFilters/dfilters.py#L100) (using Scipy python library). Reference paper: [Francisco Perdigón Romero, Liset Vázquez Romaguera, Carlos Román Vázquez-Seisdedos, Marly Guimarães Fer-nandes Costa, João Evangelista Neto, et al.  Baseline wander removal methods for ecg signals: A comparativestudy.arXiv preprint arXiv:1807.11359, 2018.](https://arxiv.org/pdf/1807.11359.pdf)
* [Deep recurrent neural networks (DRRN)](https://github.com/fperdigon/DeepFilter/blob/master/deepFilter/dl_models.py#L511). Reference paper: [Karol Antczak. Deep recurrent neural networks for ecg signal denoising.arXiv preprint arXiv:1807.11551, 2018](https://arxiv.org/pdf/1807.11551.pdf)
* [Full Convolutional Net Denoinsing Autoencoders (FCN-DAE)](https://github.com/fperdigon/DeepFilter/blob/master/deepFilter/dl_models.py#L386). Reference paper: [Hsin-Tien Chiang, Yi-Yen Hsieh, Szu-Wei Fu, Kuo-Hsuan Hung, Yu Tsao, and Shao-Yi Chien. Noise reduction in ecg signals using fully convolutional denoising autoencoders.IEEE Access, 7:60806–60813, 2019.](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8693790)

The proposed approach yields the best results on four similarity metrics: the sum of squared distance, maximum absolute square, percentage of root distance, and cosine 
similarity.

## Results

The following table present the quantitative results of DeepFilter Net compared on the same test set with other SOTA 
methods.
![Results table](ReadmeImg/results_table.png "Results table")

Qualitative results

The figure shows a portion of sele0106 ECG signal.
![Original ECG signal](ReadmeImg/fig_sele0106_orig.png "Original ECG signal")

Original ECG + BLW noise from the NSTDB.
![Original ECG signal + BLW noise](ReadmeImg/fig_sele0106_orig+blw.png "Original ECG signal + BLW noise")

The blue line is the ECG filtered using our approach.Metric values are also included.
![ECG filtered using DeepFilter](ReadmeImg/fig_sele0106_dl_filter.png "ECG filtered using DeepFilter")

The brown signal is the ECG recovered using the IIR filter, this image was included for visual comparison purposes. Metric values are also included.
![ECG filtered using IIR classical filter](ReadmeImg/fig_sele0106_iir_filter.png "ECG filtered using IIR classical filter")


## Complete Setup and Execution Guide

### Step 1: Install Miniconda3

First, you need to install Miniconda3 to manage Python environments:

**Windows:**
1. Download Miniconda3 from [https://docs.conda.io/projects/miniconda/en/latest/](https://docs.conda.io/projects/miniconda/en/latest/)
2. Run the installer and follow the installation wizard
3. Open Anaconda Prompt (or restart your terminal)

**Linux/macOS:**
```bash
# Download and install Miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow the installation prompts and restart your terminal
```

### Step 2: Clone Repository

Clone this repository to your local machine:
```bash
git clone https://github.com/EdwardJR/DeepFilter.git
cd DeepFilter
```

### Step 3: Create GPU Environment

Create the conda environment with GPU support for faster training:

```bash
conda env create -f environment_gpu.yaml
conda activate DeepFilter-GPU
```

### Step 4: Download Dataset

Download the required ECG datasets:

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy Bypass -File '.\download_data.ps1'
```

**Linux/macOS:**
```bash
bash ./download_data.sh
```

### Step 5: Train Models

Run the main training script to train all deep learning models:

```bash
python DeepFilter_main.py
```

This will train 6 different models:
- Multibranch LANLD (proposed method)
- Multibranch LANL
- Vanilla L
- Vanilla NL
- DRNN (Deep Recurrent Neural Network)
- FCN-DAE (Fully Convolutional Denoising Autoencoder)

### Step 6: Export Models

Export all trained models to TensorFlow SavedModel format:

```bash
python export_all_models.py
```

### Step 7: Convert to ONNX

Convert all models to ONNX format for cross-platform deployment:

```bash
python convert_to_onnx.py
```

### Step 8: Create Performance Visualizations

Generate comprehensive performance comparison visualizations:

```bash
python create_visual_comparison.py
```

This creates multiple visualization sets comparing all methods against classical filters.

### Step 9: Generate SNR Analysis

Create detailed Signal-to-Noise Ratio analysis:

```bash
python analyze_snr_performance.py
```


## Expected Results

After completing all steps, you will have:

- **Trained Models**: 6 deep learning models in HDF5, SavedModel, and ONNX formats
- **Performance Metrics**: Comprehensive comparison showing ~12x improvement over classical filters
- **SNR Analysis**: Demonstrating 17.37 dB SNR improvement (2.2x better than classical methods)
- **Visualizations**: Multiple comparison plots and performance charts
- **Export Formats**: Models ready for deployment in various frameworks

## GPU Requirements

For optimal performance with GPU acceleration:
- NVIDIA GPU with CUDA Compute Capability 3.5 or higher
- CUDA 11.2+ and cuDNN 8.1+
- At least 4GB GPU memory

If you encounter GPU issues, the code will automatically fall back to CPU training (slower but functional).   

## Citing DeepFilter

When citing DeepFilter please use this BibTeX entry:
   
    @article{romero2021deepfilter,
    title={DeepFilter: an ECG baseline wander removal filter using deep learning techniques},
    author={Romero, Francisco P and Pi{\~n}ol, David C and V{\'a}zquez-Seisdedos, Carlos R},
    journal={Biomedical Signal Processing and Control},
    volume={70},
    pages={102992},
    year={2021},
    publisher={Elsevier}
    }
    
## License

The MIT License (MIT)

Copyright (c) 2021 Francisco Perdigon Romero, David Castro Piñol

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
