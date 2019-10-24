# Strongeryolo-pytorch 

## Introduction
Pytorch implementation of [Stronger-Yolo](https://github.com/Stinky-Tofu/Stronger-yolo) with channel-pruning.

## Environment
python3.6, pytorch1.2(1.0+ should be ok)

## Quick Start
1 . run the following command to start training, see [yacs](https://github.com/rbgirshick/yacs) for more instructions.  
```
python main.py --config-file configs/voc.yaml  EXPER.experiment_name voc_512 devices 0,
```
2 . run the following command to test
```
python main.py --config-file configs/voc.yaml EXPER.resume best  do_test True EXPER.experiment_name voc_512 devices 0,1,
```
## Model Pruning
1 . training with sparse regularization
```
python main.py --config-file configs/voc.yaml  EXPER.experiment_name voc_512_sparse Prune.sparse True Prune.sr 0.01  
```
2 . Pruning and Finetune, check [MobileV2 Pruning](https://github.com/wlguan/MobileNet-v2-pruning) for a simplified example.
```
python main_prune.py --config-file configs/voc_prune.yaml  EXPER.experiment_name voc_512_sparse Prune.sparse True Prune.pruneratio 0.3   
```
## Transfer back to Tensorflow and make it portable.
Check [MNN-yolov3](https://github.com/wlguan/MNN-yolov3).

## Performance on VOC2007 Test(mAP)
|Model| MAP | Flops(G)| Params(M)|
| ------ | ------ | ------ | ------ |
Yolov3| 0.765|4.33|6.775|
Yolov3-sparsed| 0.750|4.33|6.775|
Yolov3-Pruned(35% pruned) |0.746 |3.00|2.815|

Note:  
1.All experiments are trained for 60 epochs.  
2.All experiments tested with threshold 0.1 in 512 resolution.

## Reference
[Stronger-Yolo](https://github.com/Stinky-Tofu/Stronger-yolo)
