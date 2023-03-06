# Locate-Globally-Segment-locally-A-Progressive-Architecture-With-Knowledge-Review-Network-for-SOD
!!!2021-7-3. We have corrected some errors. The pre-trained SGL-KRN model and PA-KRN model will be released soon... 

!!!2021-8-12. The pre-trained SGL-KRN model and PA-KRN model have been released.

### This repository is the official implementation of PA-KRN and SGL-KRN, which is proposed in "Locate Globally, Segment locally: A Progressive Architecture With Knowledge Review Network for Salient Object Detection." [PDF](https://ojs.aaai.org/index.php/AAAI/article/view/16408)

![image](https://user-images.githubusercontent.com/42328490/109591578-ba656100-7b48-11eb-8419-d258e20ed9d0.png)

## Prerequisites
- Python 3.6
- PyTorch 1.0.0
- torchvision
- Opencv
- numpy
- scipy

## Usage

### 1. Install body-atttention sampler related tools (MobulaOP)
```bash
# Clone the project
git clone https://github.com/wkcn/MobulaOP

# Enter the directory
cd MobulaOP

# Install MobulaOP
pip install -v -e .
```

### 2. Clone the repository
```
git clone https://github.com/bradleybin/Locate-Globally-Segment-locally-A-Progressive-Architecture-With-Knowledge-Review-Network-for-SOD
```
The directory shows as follow:
```bash
├─demo
│   ├── attention_sampler
│   ├── data
│   ├── dataset
│   ├── networks
│   ├── results
│   ├── KRN.py
│   ├── KRN_edge.py
│   ├── main_clm.py
│   ├── main_fsm.py
│   ├── main_joint.py
│   ├── main_SGL_KRN.py
│   ├── Solver_clm.py
│   ├── Solver_fsm.py
│   └── Solver_joint.py
├── MobulaOP
```

### 3. Download datasets
Download the ` DUTS`  and other datasets and unzip them into `demo/data` folder. (Refer to [PoolNet repository](https://github.com/backseason/PoolNet))

The directory shows as follow:
```bash
├─DUTS
│        └── DUTS-TR
│                  ├── DUTS-TR-Image
│                  ├── DUTS-TR-Mask
│                  └── DUTS-TR-Edge
├─DUTS-TE
│        ├── Imgs
│        └── test.lst
├─PASCALS
│        ├── Imgs
│        └── test.lst
├─DUTOMRON
│        ├── Imgs
│        └── test.lst
├─HKU-IS
│        ├── Imgs
│        └── test.lst
└─ECSSD
         ├── Imgs
         └── test.lst
```
### 4. Download Pretrained ResNet-50 Model for backbone
Download ResNet-50 pretrained models [Google Drive](https://drive.google.com/drive/folders/1Q2Fg2KZV8AzNdWNjNgcavffKJBChdBgy) and save it into `demo/dataset/pretrained` folder.

### 5. Train
#### 5.1 SGL-KRN

```
cd demo
python main_SGL_KRN.py
```
After training, the resulting model will be stored under results/sgl_krn/run-* folder.

#### 5.2 PA-KRN
The whole system can be trained in an end-to-end manner. To get finer results, we first train CLM and FSM sequentially and then combine them to fine-tune. 
```
cd demo
```
1. Train CLM. 
```
python main_clm.py
```
After training, the resulting model will be stored under results/clm/run-* folder.

2. Train FSM. 
```
python main_fsm.py  --clm_model path/to/pretrained/clm/folder/
```
After training, the resulting model will be stored under results/fsm/run-* folder, and * changes accordingly. 'path/to/pretrained/clm/folder/' is the path to pretrained clm folder.

3. Train PA-KRN. 
```
python main_joint.py  --clm_model path/to/pretrained/clm/folder/  --fsm_model path/to/pretrained/fsm/folder/
```
After training, the resulting model will be stored under results/joint/run-* folder. 'net_\*.pth' is the parameter of CLM model and '.pth' is the parameter of FSM model.


### 6. Test
Download pretrained SGL-KRN and PA-KRN models [Google Drive](https://drive.google.com/file/d/1v0syP-e4VkT0oJPdQ0C0SVH9VDioyG9P/view?usp=sharing).

#### 6.1 SGL-KRN
For DUTS-TE dataset testing.
```
python main_SGL_KRN.py --mode test --test_model path/to/pretrained/SGL_KRN/folder/ --test_fold path/to/test/folder/ --sal_mode t
```
'sal_mode' of ECSSD, PASCALS, DUT-OMRON, and HKU-IS are 'e', 'p', 'd', and 'h', respectively.

#### 6.2 PA-KRN
For DUTS-TE dataset testing.
```
python main_joint.py --mode test --clm_model path/to/pretrained/clm/folder/  --fsm_model path/to/pretrained/fsm/folder/ --test_fold path/to/test/folder/ --sal_mode t
```
'sal_mode' of ECSSD, PASCALS, DUT-OMRON, and HKU-IS are 'e', 'p', 'd', and 'h', respectively.


### 7. Saliency maps
We provide the pre-computed saliency maps from our paper [Google Drive](https://drive.google.com/drive/folders/1crvlMRp5oBNHs3zJ9kEYJREfw4ZjxnQm?usp=sharing) | [Baidu Disk](https://pan.baidu.com/s/1pKE4K8bckxgvttO4rgjEBw) (pwd: 9wxg).

Thanks to [PoolNet repository](https://github.com/backseason/PoolNet) and [AttentionSampler repository](https://github.com/wkcn/AttentionSampler).


### Citing PAKRN
Please cite with the following Bibtex code:

```
@inproceedings{xu2021locate,
  title={Locate Globally, Segment Locally: A Progressive Architecture With Knowledge Review Network for Salient Object Detection},
  author={Xu, Binwei and Liang, Haoran and Liang, Ronghua and Chen, Peng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={4},
  pages={3004--3012},
  year={2021}
}
```
