# Comparative Analysis of Object Detection Models for Urban Environments

## Introduction

### Objective

To compare three popular object detection models in terms of their effectiveness and efficiency in urban environments for self-driving applications. Each model will be evaluated for its suitability in detecting objects commonly encountered in urban areas, such as vehicles, pedestrians, traffic signs, and obstacles.

### Models

EfficientDet D1 640x640
SSD MobileNet V2 FPNLite 640x640
Faster R-CNN with ResNet-50 (v1) 640x640

## Model Overview

| Model                          | Architecture               | Input Size | Backbone     | Key Characteristics                 |
| ------------------------------ | -------------------------- | ---------- | ------------ | ----------------------------------- |
| EfficientDet D1                | EfficientDet               | 640x640    | EfficientNet | Good accuracy with moderate speed   |
| SSD MobileNet V2 FPNLite       | Single Shot Multibox (SSD) | 640x640    | MobileNet V2 | High speed, optimized for real-time |
| Faster R-CNN with ResNet-50 v1 | Region-based CNN (R-CNN)   | 640x640    | ResNet-50    | High accuracy, but slower inference |

## Evaluation Criteria for Urban Object Detection

### Presision

| Model                          | mAP    | mAP(large) | mAP(medium) | mAP(small) | mAP@0.5IoU | mAP@0.75IoU |
| ------------------------------ | ------ | ---------- | ----------- | ---------- | ---------- | ----------- |
| EfficientDet D1                | 0.0549 | 0.2371     | 0.2578      | 0.02       | 0.1387     | 0.026       |
| SSD MobileNet V2 FPNLite       | 0.0675 | 0.3975     | 0.2492      | 0.027      | 0.1418     | 0.0579      |
| Faster R-CNN with ResNet-50 v1 | 0.059  | 0.58       | 0.2007      | 0.0202     | 0.1266     | 0.0474      |

**Insights**:

- SSD MobileNet V2 FPNLite achieves the highest mAP for larger objects, making it suitable for detecting vehicles or other prominent objects.
- Faster R-CNN with ResNet-50 scores the highest mAP on larger objects but has relatively lower mAP on medium and small objects, indicating strong detection on prominent features in urban scenes.
- EfficientDet D1 provides balanced mAP across medium and large objects but shows lower performance on smaller objects.

### Recall

| Model                          | AR@1   | AR@10  | AR@100 | AR@100(large) | AR@100(medium) | AR@100(small) |
| ------------------------------ | ------ | ------ | ------ | ------------- | -------------- | ------------- |
| EfficientDet D1                | 0.0156 | 0.0681 | 0.0962 | 0.3738        | 0.3826         | 0.0401        |
| SSD MobileNet V2 FPNLite       | 0.0163 | 0.0751 | 0.1064 | 0.4923        | 0.3463         | 0.0611        |
| Faster R-CNN with ResNet-50 v1 | 0.0328 | 0.0947 | 0.1256 | 0.6234        | 0.3193         | 0.0848        |

**Insights**:

- Faster R-CNN with ResNet-50 shows superior average recall across most object sizes, particularly for larger objects (AR@100 large), making it robust for high-recall detection in urban settings.
- SSD MobileNet V2 FPNLite has the highest recall on small objects compared to EfficientDet D1, which makes it beneficial for real-time detection of varied object sizes in urban environments.
- EfficientDet D1 has consistent recall scores but performs lower overall compared to the other two models.

### Loss

| Model                          | Classification Loss | Localization Loss | Regularization Loss | Total Loss | RPNLoss/Localization Loss | RPNLoss/Objectness Loss |
| ------------------------------ | ------------------- | ----------------- | ------------------- | ---------- | ------------------------- | ----------------------- |
| EfficientDet D1                | 0.6827              | 0.0273            | 0.0307              | 0.7407     | 0.3826                    | 0.0401                  |
| SSD MobileNet V2 FPNLite       | 0.5404              | 0.4832            | 0.1487              | 1.1722     | 0.3463                    | 0.0611                  |
| Faster R-CNN with ResNet-50 v1 | 0.1725              | 0.2452            | 0.0                 | 1.2087     | 0.3193                    | 0.0848                  |

**Insights**:

- EfficientDet D1 has the lowest total loss among the models, particularly due to low localization and regularization losses, indicating strong potential for balanced training with low overfitting.
- SSD MobileNet V2 FPNLite shows a relatively high localization loss, which may affect its performance on spatial accuracy, though its speed advantage compensates for this in real-time applications.
- Faster R-CNN with ResNet-50 maintains low classification and localization losses, though it has a higher total loss, potentially due to a more complex architecture aimed at precise, high-resolution detections.

### Learning Rate

| Model                          | Learning Rate |
| ------------------------------ | ------------- |
| EfficientDet D1                | 0.0642        |
| SSD MobileNet V2 FPNLite       | 0.0799        |
| Faster R-CNN with ResNet-50 v1 | 0.04          |

**Insights**:

- SSD MobileNet V2 FPNLite has the highest learning rate (0.0799), which can contribute to faster convergence but may require careful monitoring to avoid overshooting optimal values.
- EfficientDet D1 has a moderate learning rate (0.0642), providing balanced training dynamics.
- Faster R-CNN with ResNet-50 has the lowest learning rate (0.04), suitable for fine-tuning given its complex architecture, which might benefit from slower, more gradual learning.

### Side by side visualization

| Model                          | 0                                                                       |
| ------------------------------ | ----------------------------------------------------------------------- |
| EfficientDet D1                | ![Side_by_side_0_0](Model_1(efficientdet)/eval/eval_side_by_side_0_0.png) |
| SSD MobileNet V2 FPNLite       | ![Side_by_side_0_0](Model_2(mobilenet)/eval/eval_side_by_side_0_0.png)    |
| Faster R-CNN with ResNet-50 v1 | ![Side_by_side_0_0](Model_3(resnet50)/eval/eval_side_by_side_0_0.png)     |

| Model                          | 1                                                                       |
| ------------------------------ | ----------------------------------------------------------------------- |
| EfficientDet D1                | ![Side_by_side_1_0](Model_1(efficientdet)/eval/eval_side_by_side_1_0.png) |
| SSD MobileNet V2 FPNLite       | ![Side_by_side_1_0](Model_2(mobilenet)/eval/eval_side_by_side_1_0.png)    |
| Faster R-CNN with ResNet-50 v1 | ![Side_by_side_1_0](Model_3(resnet50)/eval/eval_side_by_side_1_0.png)     |

| Model                          | 2                                                                       |
| ------------------------------ | ----------------------------------------------------------------------- |
| EfficientDet D1                | ![Side_by_side_2_0](Model_1(efficientdet)/eval/eval_side_by_side_2_0.png) |
| SSD MobileNet V2 FPNLite       | ![Side_by_side_2_0](Model_2(mobilenet)/eval/eval_side_by_side_2_0.png)    |
| Faster R-CNN with ResNet-50 v1 | ![Side_by_side_2_0](Model_3(resnet50)/eval/eval_side_by_side_2_0.png)     |

| Model                          | 3                                                                       |
| ------------------------------ | ----------------------------------------------------------------------- |
| EfficientDet D1                | ![Side_by_side_3_0](Model_1(efficientdet)/eval/eval_side_by_side_3_0.png) |
| SSD MobileNet V2 FPNLite       | ![Side_by_side_3_0](Model_2(mobilenet)/eval/eval_side_by_side_3_0.png)    |
| Faster R-CNN with ResNet-50 v1 | ![Side_by_side_3_0](Model_3(resnet50)/eval/eval_side_by_side_3_0.png)     |

| Model                          | 4                                                                       |
| ------------------------------ | ----------------------------------------------------------------------- |
| EfficientDet D1                | ![Side_by_side_4_0](Model_1(efficientdet)/eval/eval_side_by_side_4_0.png) |
| SSD MobileNet V2 FPNLite       | ![Side_by_side_4_0](Model_2(mobilenet)/eval/eval_side_by_side_4_0.png)    |
| Faster R-CNN with ResNet-50 v1 | ![Side_by_side_4_0](Model_3(resnet50)/eval/eval_side_by_side_4_0.png)     |

| Model                          | 5                                                                       |
| ------------------------------ | ----------------------------------------------------------------------- |
| EfficientDet D1                | ![Side_by_side_5_0](Model_1(efficientdet)/eval/eval_side_by_side_5_0.png) |
| SSD MobileNet V2 FPNLite       | ![Side_by_side_5_0](Model_2(mobilenet)/eval/eval_side_by_side_5_0.png)    |
| Faster R-CNN with ResNet-50 v1 | ![Side_by_side_5_0](Model_3(resnet50)/eval/eval_side_by_side_5_0.png)     |

| Model                          | 6                                                                       |
| ------------------------------ | ----------------------------------------------------------------------- |
| EfficientDet D1                | ![Side_by_side_6_0](Model_1(efficientdet)/eval/eval_side_by_side_6_0.png) |
| SSD MobileNet V2 FPNLite       | ![Side_by_side_6_0](Model_2(mobilenet)/eval/eval_side_by_side_6_0.png)    |
| Faster R-CNN with ResNet-50 v1 | ![Side_by_side_6_0](Model_3(resnet50)/eval/eval_side_by_side_6_0.png)     |

| Model                          | 7                                                                       |
| ------------------------------ | ----------------------------------------------------------------------- |
| EfficientDet D1                | ![Side_by_side_7_0](Model_1(efficientdet)/eval/eval_side_by_side_7_0.png) |
| SSD MobileNet V2 FPNLite       | ![Side_by_side_7_0](Model_2(mobilenet)/eval/eval_side_by_side_7_0.png)    |
| Faster R-CNN with ResNet-50 v1 | ![Side_by_side_7_0](Model_3(resnet50)/eval/eval_side_by_side_7_0.png)     |

| Model                          | 8                                                                       |
| ------------------------------ | ----------------------------------------------------------------------- |
| EfficientDet D1                | ![Side_by_side_8_0](Model_1(efficientdet)/eval/eval_side_by_side_8_0.png) |
| SSD MobileNet V2 FPNLite       | ![Side_by_side_8_0](Model_2(mobilenet)/eval/eval_side_by_side_8_0.png)    |
| Faster R-CNN with ResNet-50 v1 | ![Side_by_side_8_0](Model_3(resnet50)/eval/eval_side_by_side_8_0.png)     |

| Model                          | 9                                                                       |
| ------------------------------ | ----------------------------------------------------------------------- |
| EfficientDet D1                | ![Side_by_side_9_0](Model_1(efficientdet)/eval/eval_side_by_side_9_0.png) |
| SSD MobileNet V2 FPNLite       | ![Side_by_side_9_0](Model_2(mobilenet)/eval/eval_side_by_side_9_0.png)    |
| Faster R-CNN with ResNet-50 v1 | ![Side_by_side_9_0](Model_3(resnet50)/eval/eval_side_by_side_9_0.png)     |

## Output Result

### EfficientDet D1

![Loss](Model_1(efficientdet)/train/loss.png)

![steps_per_sec](Model_1(efficientdet)/train/steps_per_sec.png)

![steps_per_sec](Model_3(resnet50)/output.gif)

### SSD MobileNet V2 FPNLite

![Loss](Model_2(mobilenet)/train/loss.png)

![steps_per_sec](Model_2(mobilenet)/train/steps_per_sec.png)

![steps_per_sec](Model_2(mobilenet)/output.gif)

### Faster R-CNN with ResNet-50 v1

![Loss](Model_3(resnet50)/train/loss.png)

![steps_per_sec](Model_3(resnet50)/train/steps_per_sec.png)

![steps_per_sec](Model_3(resnet50)/output.gif)
