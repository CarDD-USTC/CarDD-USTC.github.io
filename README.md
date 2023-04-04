# CarDD: A New Dataset for Vision-based Car Damage Detection

by Xinkuang Wang, Wenjing Li, Zhongcheng Wu.



# CarDD Dataset

**Information about the CarDD dataset is available at [https://cardd-ustc.github.io/](https://cardd-ustc.github.io/).**


# Car Damage Detection and Segmentation

### Environment setup

Environment requirement: Pytorch 1.7.0 + Python 3.8 + CUDA 11.0

1. Clone the repo:

    ```
    git clone https://github.com/CarDD-USTC/CarDD-USTC.github.io.git
	cd CarDD-USTC.github.io/code/CarDD_detection
    ```

2. Prepare the environment:

    ```
    pip install openmim
    mim install mmdet
    pip install mmcv==1.7.0
    export MPLBACKEND='Agg' && export PYTHONPATH=$(CODE_PATH)/CarDD_detection/
    ```

### Usage
1. Download CarDD at [https://cardd-ustc.github.io/](https://cardd-ustc.github.io/).
   Download pretrained models at [Model Zoo](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/model_zoo.md) to $(MODEL_PATH).
   
 
2. Train:
    ```
    python tools/train.py configs/car_damage/DCN_plus_cfg.py --work-dir $(WORK_PATH)
    ```

3. Test:
    ```
    python tools/test.py configs/car_damage/DCN_plus_cfg.py $(WORK_PATH)/epoch_24.pth --eval bbox segm --options "classwise=True"
    ```

4. Test and visualize:
    ```
    python tools/test.py configs/car_damage/DCN_plus_cfg.py $(WORK_PATH)/epoch_24.pth --show-dir $(VIS_PATH) --show-score-thr 0.7
    ```
   
5. Only inference:
    ```
    python tools/inference.py \
    --img-path=$(IMG_PATH) \
    --save-path=$(SAVE_PATH) \
    --config-file=configs/car_damage/DCN_plus_cfg.py  \
    --checkpoint-file=$(WORK_PATH)/epoch_24.pth
    ```

# Salient Damage Detection

### Environment setup

Please refer to each repository:

[U2Net](https://github.com/xuebinqin/U-2-Net) 
| [PoolNet](https://github.com/backseason/PoolNet) 
| [KRN](https://github.com/bradleybin/Locate-Globally-Segment-locally-A-Progressive-Architecture-With-Knowledge-Review-Network-for-SOD) 
| [CSNet](https://github.com/ShangHua-Gao/SOD100K)
| [Saliency-Evaluation-Toolbox](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox)

### Usage
- U2Net:
    ```
    cd $(CODE_PATH)/CarDD_SOD/U2-Net/
    train: python u2net_train.py
    test: python u2net_test.py
    ```

- PoolNet:
    ```
    cd $(CODE_PATH)/CarDD_SOD/PoolNet/
    train: python main.py --arch resnet --mode train --device 0 --data_root $(DATA_PATH)/CarDD_SOD/ --save_folder $(WORK_PATH)
    test: python main.py --mode test --model $(WORK_PATH)/run-0/models/final.pth --test_fold $(SAVE_PATH) --data_root $(DATA_PATH)
    ```
  
- KRN:
    ```
    cd $(CODE_PATH)/CarDD_SOD/KRN/
    train: python main_SGL_KRN.py --mode train --device 0 --data_root $(DATA_PATH)/CarDD_SOD/ --save_folder $(WORK_PATH)
    test: python main_SGL_KRN.py --mode test --device 0 --sal_mode t --test_model $(WORK_PATH)/run-0/models/final.pth --test_fold $(SAVE_PATH) --data_root $(DATA_PATH)
    ```
  
- CSNet:
    ```
    cd $(CODE_PATH)/CarDD_SOD/CSNet/CSNet_training/
    train: python train.py --config configs/csnet-L-x2_train-CarDD.yml
    test: python test.py --config configs/csnet-L-x2_train-CarDD.yml
    ```

- Evaluate:
    
    Please refer to [Saliency-Evaluation-Toolbox](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox).

    
## Acknowledgments

* Segmentation code and models are from [mmdetection](https://github.com/open-mmlab/mmdetection).
* SOD code and models are respectively from [U2Net](https://github.com/xuebinqin/U-2-Net) 
| [PoolNet](https://github.com/backseason/PoolNet) 
| [KRN](https://github.com/bradleybin/Locate-Globally-Segment-locally-A-Progressive-Architecture-With-Knowledge-Review-Network-for-SOD) 
| [CSNet](https://github.com/ShangHua-Gao/SOD100K)
| [Saliency-Evaluation-Toolbox](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox).


## Citation
If you found this code helpful, please consider citing: 
```
@article{cardd,
  author={Wang, Xinkuang and Li, Wenjing and Wu, Zhongcheng},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={CarDD: A New Dataset for Vision-Based Car Damage Detection}, 
  year={2023},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TITS.2023.3258480}
}
```
