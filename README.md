# Strongeryolo-pytorch 

## Introduction
This project is inspired by [Stronger-Yolo](https://github.com/Stinky-Tofu/Stronger-yolo). I reimplemented with Pytorch and continue improving yolov3 with latest papers.  
This project will also try out some model-compression approaches(e.g. channel-pruning). 

## Environment
python3.6, pytorch1.2(1.0+ should be ok), ubuntu14/16/18 tested.

## Quick Start
1 . run the following command to start training, see [yacs](https://github.com/rbgirshick/yacs) for more instructions.  
```
python main.py --config-file configs/strongerv3.yaml  EXPER.experiment_name voc_512 devices 0,
```
2 . run the following command to test
```
python main.py --config-file configs/strongerv3.yaml EXPER.resume best  do_test True EXPER.experiment_name voc_512 devices 0,1,
```
## Improvement with latest papers(Using StrongerV3 as baseline)
|model|mAP50|mAP75|configs|
| ------ | ------ | ------ |------ |
|baseline(with GIOU)|0.765 |0.391|voc.yaml|
|+ [focal loss](https://arxiv.org/abs/1708.02002)|0.772|0.438 |strongerv3_clsfocal.yaml|
|+ [kl loss](https://github.com/yihui-he/KL-Loss)|0.778|0.449 |strongerv3_kl.yaml|
|+ [var vote](https://github.com/yihui-he/KL-Loss)|0.781|0.464 |strongerv3_kl.yaml|

Note:  
1.Set EVAL.varvote=True to enable varvote in KL-loss. 
## Model Pruning
1 . training with sparse regularization
```
python main.py --config-file configs/strongerv3_sparse.yaml  EXPER.experiment_name voc_512_sparse Prune.sparse True Prune.sr 0.01  
```
2 . Pruning and Finetune, check [MobileV2 Pruning](https://github.com/wlguan/MobileNet-v2-pruning) for a simplified example.
```
python main_prune.py --config-file configs/strongerv3_prune.yaml  EXPER.experiment_name voc_512_sparse Prune.sparse True Prune.pruneratio 0.35   
```
### Transfer back to Tensorflow and make it portable.
Check [MNN-yolov3](https://github.com/wlguan/MNN-yolov3).

### Performance on VOC2007 Test(mAP) after pruning
|Model| MAP | Flops(G)| Params(M)|
| ------ | ------ | ------ | ------ |
Yolov3| 0.765|4.33|6.775|
Yolov3-sparsed| 0.750|4.33|6.775|
Yolov3-Pruned(35% pruned) |0.746 |3.00|2.815|

Note:  
1.All experiments are trained for 60 epochs.  
2.All experiments tested with threshold 0.1 in 512 resolution.
## Supported backbone
- [x] MobileV2
- [x] DarkNet  
...
## Reference
[Stronger-Yolo](https://github.com/Stinky-Tofu/Stronger-yolo)  
[focal-loss](https://arxiv.org/abs/1708.02002)  
[kl-loss](https://github.com/yihui-he/KL-Loss)
