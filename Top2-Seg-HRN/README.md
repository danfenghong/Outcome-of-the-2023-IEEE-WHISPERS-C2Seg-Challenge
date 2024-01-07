# Top2: Multimodal unsupervised domain adaptation for remote sensing image segmentation

Take AB dataset as example:
## 1 Train/test the model using HSI data
```
python hsi_crop_trainval.py --mode train --config "/Top2-Seg-HRN/utils/train_cfg/AB_HRNet48.yaml"
python hsi_crop_trainval.py --mode test --config "/Top2-Seg-HRN/utils/train_cfg/AB_HRNet48.yaml" --resume_path "/Top2-Seg-HRN/run/train/xxx/best_model.pt"
```
## 2 Train/test the model using MSI data
```
python msi_crop_trainval.py --mode train --config "/Top2-Seg-HRN/utils/train_cfg/AB_HRNet48.yaml"
python msi_crop_trainval.py --mode test --config "/Top2-Seg-HRN/utils/train_cfg/AB_HRNet48.yaml" --resume_path "/Top2-Seg-HRN/run/train/xxx/best_model.pt"
```
## 3 Train/test the model using SAR data
```
python sar_crop_trainval.py --mode train --config "/Top2-Seg-HRN/utils/train_cfg/AB_HRNet48.yaml"
python sar_crop_trainval.py --mode test --config "/Top2-Seg-HRN/utils/train_cfg/AB_HRNet48.yaml" --resume_path "/Top2-Seg-HRN/run/train/xxx/best_model.pt"
```
## 4 Train/test the model using multi-modal data
```
python crop_trainval.py --mode train --config "/Top2-Seg-HRN/utils/train_cfg/AB_HRNet48.yaml"
python crop_trainval.py --mode test --config "/Top2-Seg-HRN/utils/train_cfg/AB_HRNet48.yaml" --resume_path "/Top2-Seg-HRN/run/train/xxx/best_model.pt"
```

## 5 Fuse multiple models to improve performance
```
python four_multi_model_fusion_trainval.py --mode test
```

## 6 Multimodal unsupervised domain adaptation
### 6.1 Pre-train
```
python uda_crop_trainval.py --mode uda_train_pretrained --config "/Top2-Seg-HRN/utils/train_cfg/AB_uda_seghrnet_pretrained.yaml"
```
### 6.2 Fine-tune
```
python uda_crop_trainval.py --mode uda_train_da --config "/Top2-Seg-HRN/utils/train_cfg/AB_uda_seghrnetocr_da.yaml" --resume_path "/Top2-Seg-HRN/run/uda_train_pretrained/xxx/best_model.pt"
```
### 6.3 Test
```
python uda_crop_trainval.py --mode uda_test --config "/Top2-Seg-HRN/utils/train_cfg/AB_uda_seghrnetocr_da.yaml" --resume_path "/Top2-Seg-HRN/run/uda_train_da/xxx/last_model.pt"
```
