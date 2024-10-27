# Comparative Analysis of Object Detection Models for Urban Environments

## Introduction

### Objective

To compare three popular object detection models in terms of their effectiveness and efficiency in urban environments for self-driving applications. Each model will be evaluated for its suitability in detecting objects commonly encountered in urban areas, such as vehicles, pedestrians.

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

### Precision

| Model                          | mAP    | mAP(large) | mAP(medium) | mAP(small) | mAP@0.5IoU | mAP@0.75IoU |
| ------------------------------ | ------ | ---------- | ----------- | ---------- | ---------- | ----------- |
| EfficientDet D1                | 0.0549 | 0.2371     | 0.2578      | 0.02       | 0.1387     | 0.026       |
| SSD MobileNet V2 FPNLite       | 0.0675 | 0.3975     | 0.2492      | 0.027      | 0.1418     | 0.0579      |
| Faster R-CNN with ResNet-50 v1 | 0.059  | 0.58       | 0.2007      | 0.0202     | 0.1266     | 0.0474      |

**Insights**:

- EfficientDet D1 provides balanced mAP across medium and large objects but shows lower performance on smaller objects.
  ![precision](Model_1(efficientdet)/eval/precision.png)
- SSD MobileNet V2 FPNLite achieves the highest mAP for larger objects, making it suitable for detecting vehicles or other prominent objects.
  ![precision](Model_2(mobilenet)/eval/precision.png)
- Faster R-CNN with ResNet-50 scores the highest mAP on larger objects but has relatively lower mAP on medium and small objects, indicating strong detection on prominent features in urban scenes.
  ![precision](Model_3(resnet50)/eval/precision.png)

**Summary**:

**SSD MobileNet V2 FPNLite** achieves the highest mAP for larger objects, making it effective for detecting prominent features in urban environments, while **Faster R-CNN with ResNet-50** performs well on large objects, showing potential for high-accuracy detection of vehicles and pedestrians.

### Recall

| Model                          | AR@1   | AR@10  | AR@100 | AR@100(large) | AR@100(medium) | AR@100(small) |
| ------------------------------ | ------ | ------ | ------ | ------------- | -------------- | ------------- |
| EfficientDet D1                | 0.0156 | 0.0681 | 0.0962 | 0.3738        | 0.3826         | 0.0401        |
| SSD MobileNet V2 FPNLite       | 0.0163 | 0.0751 | 0.1064 | 0.4923        | 0.3463         | 0.0611        |
| Faster R-CNN with ResNet-50 v1 | 0.0328 | 0.0947 | 0.1256 | 0.6234        | 0.3193         | 0.0848        |

**Insights**:

- EfficientDet D1 has consistent recall scores but performs lower overall compared to the other two models.
  ![recall](Model_1(efficientdet)/eval/recall.png)
- SSD MobileNet V2 FPNLite has the highest recall on small objects compared to EfficientDet D1, which makes it beneficial for real-time detection of varied object sizes in urban environments.
  ![recall](Model_2(mobilenet)/eval/recall.png)
- Faster R-CNN with ResNet-50 shows superior average recall across most object sizes, particularly for larger objects (AR@100 large), making it robust for high-recall detection in urban settings.
  ![recall](Model_3(resnet50)/eval/recall.png)

**Summary**:

**Faster R-CNN with ResNet-50** shows superior recall across object sizes, particularly large objects, indicating robustness in high-recall scenarios, while **SSD MobileNet V2 FPNLite** balances recall for small and large objects, making it a good candidate for varied urban detection needs.

### Loss

| Model                          | Classification Loss | Localization Loss | Regularization Loss | Total Loss | RPNLoss/Localization Loss | RPNLoss/Objectness Loss |
| ------------------------------ | ------------------- | ----------------- | ------------------- | ---------- | ------------------------- | ----------------------- |
| EfficientDet D1                | 0.6827              | 0.0273            | 0.0307              | 0.7407     | -                         | -                       |
| SSD MobileNet V2 FPNLite       | 0.5404              | 0.4832            | 0.1487              | 1.1722     | -                         | -                       |
| Faster R-CNN with ResNet-50 v1 | 0.1725              | 0.2452            | 0.0                 | 1.2087     | 0.6875                    | 0.1035                  |

**Insights**:

- EfficientDet D1 has the lowest total loss among the models, particularly due to low localization and regularization losses, indicating strong potential for balanced training with low over-fitting.
  ![Loss](Model_1(efficientdet)/eval/loss.png)
- SSD MobileNet V2 FPNLite shows a relatively high localization loss, which may affect its performance on spatial accuracy, though its speed advantage compensates for this in real-time applications.
  ![Loss](Model_2(mobilenet)/eval/loss.png)
- Faster R-CNN with ResNet-50 maintains low classification and localization losses, though it has a higher total loss, potentially due to a more complex architecture aimed at precise, high-resolution detections.
  ![Loss](Model_3(resnet50)/eval/loss.png)

**Summary**:

**EfficientDet D1** achieves the lowest total evaluation loss, suggesting a balanced performance with minimal over-fitting, while **SSD MobileNet V2 FPNLite** has higher localization loss, which may impact spatial accuracy but supports real-time applications due to its efficient architecture.

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

## Train Criteria for Urban Object Detection

### Loss

| Model                          | Classification Loss | Localization Loss | Regularization Loss | Total Loss | RPNLoss/Localization Loss | RPNLoss/Objectness Loss |
| ------------------------------ | ------------------- | ----------------- | ------------------- | ---------- | ------------------------- | ----------------------- |
| EfficientDet D1                | 0.2617              | 0.0193            | 0.0307              | 0.3117     | -                         | -                       |
| SSD MobileNet V2 FPNLite       | 0.16                | 0.1807            | 0.1487              | 0.4894     | --                        |                         |
| Faster R-CNN with ResNet-50 v1 | 0.1964              | 0.3369            | 0.0                 | 1.0715     | 0.4714                    | 0.0668                  |

**Insights**:

- EfficientDet D1 has the lowest total training loss among the models, with particularly low localization and regularization losses.
  ![Loss](Model_1(efficientdet)/train/loss.png)
- SSD MobileNet V2 FPNLite shows a relatively high localization loss, which might impact its spatial accuracy in predicting precise bounding boxes.
  ![Loss](Model_2(mobilenet)/train/loss.png)
- Faster R-CNN with ResNet-50 exhibits moderate classification and localization losses, but its overall total loss is higher due to its complex architecture. The higher training loss reflects the model's intricate multi-stage process, which is geared towards achieving high-resolution, precise detections, albeit with a trade-off in training stability.
  ![Loss](Model_3(resnet50)/train/loss.png)

**Summary**:

**EfficientDet D1** demonstrates the lowest total training loss, indicating efficient learning with low over-fitting risk, while **Faster R-CNN with ResNet-50** shows a higher training loss, reflecting its complex architecture and focus on high-resolution, precise detections despite reduced training stability.

### Learning Rate

| Model                          | Learning Rate |
| ------------------------------ | ------------- |
| EfficientDet D1                | 0.0642        |
| SSD MobileNet V2 FPNLite       | 0.0799        |
| Faster R-CNN with ResNet-50 v1 | 0.04          |

**Insights**:

- EfficientDet D1 has a moderate learning rate (0.0642), providing balanced training dynamics.
  ![learning_rate](Model_1(efficientdet)/train/learning_rate.png)
- SSD MobileNet V2 FPNLite has the highest learning rate (0.0799), which can contribute to faster convergence but may require careful monitoring to avoid overshooting optimal values.
  ![learning_rate](Model_2(mobilenet)/train/learning_rate.png)
- Faster R-CNN with ResNet-50 has the lowest learning rate (0.04), suitable for fine-tuning given its complex architecture, which might benefit from slower, more gradual learning.
  ![learning_rate](Model_3(resnet50)/train/learning_rate.png)

**Summary**:

**EfficientDet D1** provides balanced training dynamics, while **SSD MobileNet V2 FPNLite** higher rate supports faster convergence but may require monitoring, and **Faster R-CNN with ResNet-50** benefits from a lower learning rate, allowing more gradual, stable learning.

### Steps Per Second

| Model                          | Steps Per Seconds |
| ------------------------------ | ----------------- |
| EfficientDet D1                | 1.4124            |
| SSD MobileNet V2 FPNLite       | 0.6544            |
| Faster R-CNN with ResNet-50 v1 | 5.7944            |

**Insights**:

- EfficientDet D1 has a moderate steps per second (1.4124), achieving a good balance between training speed and computational efficiency. This rate supports practical training times while maintaining model accuracy, making it suitable for real-time applications.
  ![steps_per_sec](Model_1(efficientdet)/train/steps_per_sec.png)
- SSD MobileNet V2 FPNLite has the lowest steps per second (0.6544), indicating slower data processing. While this might limit training speed, SSD MobileNet is generally optimized for fast inference, meaning it could still perform well in real-time deployment.
  ![steps_per_sec](Model_2(mobilenet)/train/steps_per_sec.png)
- Faster R-CNN with ResNet-50 v1 has the highest steps per second (5.7944), allowing for rapid data processing during training. However, this high rate might not translate directly to real-time efficiency due to the model's complex architecture and higher overall loss.
  ![steps_per_sec](Model_3(resnet50)/train/steps_per_sec.png)

**Summary**:

**EfficientDet D1** achieves a good balance between training speed and computational efficiency, **SSD MobileNet V2 FPNLite** is slower in processing steps but optimized for real-time inference, and **Faster R-CNN with ResNet-50** has the highest steps per second, though its complexity may limit real-time deployment.

## Output Result

### EfficientDet D1

![output](Model_1(efficientdet)/output.gif)

### SSD MobileNet V2 FPNLite

![output](Model_2(mobilenet)/output.gif)

### Faster R-CNN with ResNet-50 v1

![steps_per_sec](Model_3(resnet50)/output.gif)

## Conclusion

After evaluating the models based on accuracy, recall, training and evaluation losses, and inference speed, **SSD MobileNet V2 FPNLite** emerges as the most suitable choice for real-time urban object detection in self-driving applications:

- **SSD MobileNet V2 FPNLite** offers a strong balance between speed and accuracy. With the highest inference speed among the models, it is well-suited for real-time detection, which is crucial in fast-paced urban environments. Additionally, its high mAP for larger objects like vehicles ensures that it can reliably detect the most prominent objects on the road. While its localization loss is slightly higher, affecting spatial accuracy, its quick processing rate makes it ideal for applications where real-time detection is prioritized.
- **EfficientDet D1** provides balanced performance with low total loss, suggesting it would be a good choice for applications where accuracy across various object sizes is prioritized alongside moderate speed. However, it may not match SSD MobileNet V2 FPNLite in terms of inference speed.
- **Faster R-CNN with ResNet-50** is best suited for high-precision tasks where inference speed is less critical. Its robust recall and precise detections make it appropriate for high-stakes applications but may be less ideal for scenarios requiring real-time responses.

**Recommended Model**: For your specific case, **SSD MobileNet V2 FPNLite** stands out due to its real-time speed advantage, high performance on large objects, and balanced recall for smaller objects, making it the optimal choice for self-driving applications in urban environments.
