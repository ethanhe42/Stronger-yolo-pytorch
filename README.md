# Strongeryolo-pytorch 

## Introduction
This project is inspired by [Stronger-Yolo](https://github.com/Stinky-Tofu/Stronger-yolo). I reimplemented with Pytorch and continue improving yolov3 with latest papers.  
This project will also try out some model-compression approaches(e.g. channel-pruning).  
See **reimplementation results** in [MODELZOO](docs/MODELZOO.md).
## Environment
python3.6, pytorch1.2(1.0+ should be ok), ubuntu14/16/18 tested.

## Quick Start
All checkpoints as well as converted darknet can be downloaded here.[链接](https://pan.baidu.com/s/17VK455rp4B_SRhEmklT_ig) 提取码: i3pa  
**See [Usage.md](docs/Usage.md) for details.** 
## Improvement with latest papers(Using StrongerV3 as baseline)
### All results all tested with 544*544 and threshold 0.1
|model|mAP50|mAP75|configs|baseline|
| ------ | ------ | ------ |------ |------ |
|baseline(with GIOU)|79.6 |43.4|strongerv3.yaml|-|
|+ [kl loss&&varvote](https://github.com/yihui-he/KL-Loss)|78.9|49.2 |strongerv3_kl.yaml|strongerv3.yaml|  
|+ [ASFF](https://github.com/ruinmessi/ASFF)|80.6|46.6 |strongerv3_asff.yaml|strongerv3.yaml|
|+ All improvement|81.1|53.0 |strongerv3_all.yaml|strongerv3.yaml|

Note:  
1.Set EVAL.varvote=True to enable varvote in KL-loss. According to the paper, kl-loss(and varvote) can strongly boost the performance of mAP75(or higher), but decrease mAP50 slightly.  
2.The details(e.g. channel number) of ASFF module is not completely the same as the original implementation.  
3.The **All** version including other small tricks like removing relu in detection head. Check config file for details. 
## Performance on VOC2007 Test(mAP) after pruning
|Model| Backbone|MAP | Flops(G)| Params(M)|
| ------ | ------ | ------ | ------ |------ |
strongerv3| Mobilev2|79.6|4.33|6.775|
strongerv3-sparsed|Mobilev2|77.4|4.33|6.775|
strongerv3-Pruned(30% pruned) |Mobilev2|77.1 |3.14|3.36|
strongerv2| Darknet53|80.2|49.8|61.6|
strongerv2-sparsed|Darknet53|78.1|49.8|61.6|
strongerv2-Pruned(20% pruned) |Darknet53|76.8 |49.8|45.2|  

Note:  
1.Tuning _C.Prune.sr can get better prune ratio, I picked the official number 0.01.  
## Supported backbone
- [x] MobileV2(Pruning suppoted)
- [x] DarkNet(Pruning supported)
...
## Reference
[Stronger-Yolo](https://github.com/Stinky-Tofu/Stronger-yolo)  
[focal-loss](https://arxiv.org/abs/1708.02002)  
[kl-loss](https://github.com/yihui-he/KL-Loss)
[ASFF](https://github.com/ruinmessi/ASFF)
