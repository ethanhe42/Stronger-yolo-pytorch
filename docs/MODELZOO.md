# Strongeryolo-pytorch 
## Results compared with the original implementation
|ModelType|InitialWeight|mAP50|config|
| ------ | ------ | ------ |------ |
|v1-original|yolov3.weights|88.8|-|
|v2-original|darknet53.conv.74|83.3|-|
|v3-original|mobilenetv2|78.9|-|
|v1-this|yolov3.weights|86.2|strongerv1.yaml|
|v2-this|darknet53.conv.74|80.2|strongerv2.yaml|
|v3-this|mobilenetv2|79.6|strongerv3.yaml|  
|v3-this|mobilenetv2-0.75|76.97|strongerv3_0.75.yaml|  

Note: 
- This project use threshold=0.1 for faster evaluation,while the original implementation use 0.01.
- Adjust the training schedules(total epochs,lr scheduler) may further boost the performance. I pick StepLR instead of ConsinLR to accelerate training procedure. Continue training may give better results. 

## Components ported from Original Implementations.
- [x] data augmentation
- [x] multi scale train
- [x] focal loss
- [x] soft nms
- [x] mix up
- [x] label smooth
- [ ] consine learning rate(Just replace the lr scheduler.)
- [x] GIOU
- [ ] multi scale test (See another [repo](https://github.com/wlguan/pytorch-yolov3) of mine for details.)
