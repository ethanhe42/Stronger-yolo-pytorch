# Strongeryolo-pytorch 
## Results compared with the original implementation
|ModelType|InitialWeight|mAP50|config|
| ------ | ------ | ------ |------ |
|v1-original|yolov3.weights|88.8|-|
|v2-original|darknet53.conv.74|83.3|-|
|v3-original|mobilenetv2|78.9|-|
|v1-this|yolov3.weights|86.2|strongerv1.yaml|
|v2-this|darknet53.conv.74|80.2|strongerv2.yaml|
|v3-this|mobilenetv2|-|strongerv3.yaml|
Note: 
- This project use threshold=0.1 for faster evaluation,while the original implementation use 0.01.
- Adjust the training schedules(total epochs,lr scheduler) may further boost the performance. I pick StepLR instead of ConsinLR to accelerate training procedure. Continue training may give better results. 