# Strongeryolo-pytorch 

## Introduction
This project is inspired by [Stronger-Yolo](https://github.com/Stinky-Tofu/Stronger-yolo). I reimplemented with Pytorch and continue improving yolov3 with latest papers.  
This project will also try out some model-compression approaches(e.g. channel-pruning).  
See **reimplementation results** in [MODELZOO](models/MODELZOO.md).
## Environment
python3.6, pytorch1.2(1.0+ should be ok), ubuntu14/16/18 tested.

## Quick Start
1 . run the following command to start training, see [yacs](https://github.com/rbgirshick/yacs) for more instructions.  
```
python main.py --config-file configs/strongerv3.yaml  EXPER.experiment_name strongerv3 devices 0,
```
2 . run the following command to test
```
python main.py --config-file configs/strongerv3.yaml EXPER.resume best  do_test True EXPER.experiment_name strongerv3 devices 0,1,
```

## Model Pruning
1 . training with sparse regularization
```
python main.py --config-file configs/strongerv3_sparse.yaml  EXPER.experiment_name strongerv3_sparse Prune.sparse True Prune.sr 0.01  
```
2 . Pruning and Finetune, check [MobileV2 Pruning](https://github.com/wlguan/MobileNet-v2-pruning) for a simplified example.
```
python main_prune.py --config-file configs/strongerv3_prune.yaml  EXPER.experiment_name strongerv3_sparse Prune.sparse True Prune.pruneratio 0.3   
```
3 . Test the pruned model
```
python main_prune.py --config-file configs/strongerv3_prune.yaml Prune.pruneratio 0.3 Prune.do_test True   
```
### Transfer back to Tensorflow and make it portable.
Check [MNN-yolov3](https://github.com/wlguan/MNN-yolov3).  
- [ ] A pytorch->tensorflow script is underway.
