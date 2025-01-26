
# Multi-class Classification of Insects: A Focus on the Asian Hornet

## Project Overview
### What is this Project About?
- **Focus on Species Classification**: Leveraging advanced deep learning techniques for accurate species identification.
- **Relevance to Conservation Efforts**: Highlights the ecological importance of species identification.
- **Efforts to Collect Real-Time Images**: Synthesizing and gathering diverse datasets for real-world applicability.
- **Broader Scope with 8 Classes**: Includes various insect species for comprehensive analysis.

### Why This Project Held Interest for Me?
- **Fine-Grained Differences in Species**: Addressing challenges in identifying visually similar species.
- **Scarcity of Comprehensive Studies**: Bridging gaps in available research and data.
- **Dataset Challenges and Innovation**: Employing innovative approaches to create a balanced dataset.
- **Application of State-of-the-Art Techniques**: Incorporating cutting-edge deep learning methods.

---

## Dataset Creation and Preparation
### Sources of the Dataset
- **Roboflow (Website)**
- **Video Frames**
- **Web Scraping**
- **Image Augmentation Techniques**: Improving dataset diversity and robustness.

### Dataset Balancing
- **Key Benefits of Dataset Strategy**:
  - Diversity
  - Class Balance
  - Generalisation

### Selection Criteria for the Species
- **Abundance and Impact in the UK**: Focus on species like Asian Hornet, European Hornet, and Oriental Hornet.
- **Similar Appearance and Potential for Misclassification**: Addressing challenges with species like Hoverfly, Honey Bee, Carpenter Bee, Common Wasp, and Asian Giant Hornet.
- **Importance for Ecological and Public Awareness**: Enhancing awareness and research on insect biodiversity.

---

## Selected Species
- Asian Hornet
- Asian Giant Hornet
- European Hornet
- Carpenter Bee
- Oriental Hornet
- Hoverfly
- Honey Bee
- Common Wasp

---

## Objectives Achieved
- Dataset Development and Curation
- Implementation of Baseline Models
- Fine-tuning and Evaluation of Pretrained Models
- Analysis and Reporting

---

## Deep Learning Models for Insect Classification
### Custom CNN Models
- Development of Five Custom CNN Versions
- **Hyperparameter Tuning**:
  - Learning Rate
  - Batch Size
  - L2 Regularisation
  - Convolutional Layers
  - Augmentation Variations

### Transfer Learning Models (Pretrained Models Used)
- VGG16
- MobileNetV2
- EfficientNetB0
- DenseNet121
- InceptionV3
- ResNet50

### Parameter Tuning for Pretrained Models
| Augmentation Parameters | First Version | Second Version |
|--------------------------|---------------|----------------|
| Rescale                 | 1.0/255       | 1.0/255        |
| Zoom Range              | 0.3           | 0.2            |
| Width Shift Range       | 0.3           | 0.2            |
| Height Shift Range      | 0.3           | 0.2            |
| Rotation Range          | 30 degrees    | 20 degrees     |
| Brightness Range        | [0.7, 1.3]    | [0.8, 1.2]     |
| Shear Range             | 0.2           | -              |
| Horizontal Flip         | TRUE          | TRUE           |
| Fill Mode               | 'nearest'     | 'nearest'      |

---

## Impact of Augmentation Variations on CNN Performance
| Metric              | First Augmentation Variation | Second Augmentation Variation | Improvement |
|---------------------|-----------------------------|-------------------------------|-------------|
| Training Accuracy   | 61.81%                      | 73.97%                        | 12.16%      |
| Validation Accuracy | 72.17%                      | 80.27%                        | 8.10%       |
| Test Accuracy       | 74.15%                      | 80.43%                        | 6.28%       |

---

## Improving Accuracy with Custom CNN Parameters
| Parameters           | Version 2 | Version 3 | Version 4 | Version 5 |
|----------------------|-----------|-----------|-----------|-----------|
| Batch size          | 32        | 32        | 32        | 64        |
| L2 Regularization   | 0.00005   | 0.00005   | 0.001     | 0.001     |
| Learning Rate       | 0.001     | 0.0001    | 0.001     | 0.001     |
| Augmentation Strategy | Second Version | Second Version | Second Version | Second Version |
| Training Accuracy   | 73.97%    | 70.44%    | 86.01%    | 85.83%    |
| Validation Accuracy | 80.27%    | 74.38%    | 90.70%    | 83.05%    |
| Test Accuracy       | 80.43%    | 76.04%    | 86.32%    | 85.07%    |

---

## Stratified K-Fold Validation Analysis on Custom CNN
| Metric                  | Value          |
|-------------------------|----------------|
| K-Fold                 | 5              |
| Fold Accuracies        | 85.44%, 87.20%, 87.57%, 85.19%, 85.06% |
| Mean Accuracy          | 86.09%         |
| Standard Deviation     | 1.07           |
| Variance               | 1.14           |
| Coefficient of Variation (CV) | 0.0124 |

---

## Accuracies of Pre-trained Models with Optimum Parameters
| Model         | Convergence (Epochs) | Train Accuracy | Validation Accuracy | Test Accuracy |
|---------------|-----------------------|----------------|---------------------|---------------|
| VGG16         | 78                    | 94.44%         | 91.78%              | 88.08%        |
| MobileNetV2   | 28                    | 98.81%         | 96.14%              | 96.61%        |
| EfficientNetB0| 23                    | 98.28%         | 97.79%              | 97.19%        |
| DenseNet121   | 37                    | 98.77%         | 98.29%              | 97.81%        |
| InceptionV3   | 25                    | 98.95%         | 98.92%              | 98.73%        |
| ResNet50      | 24                    | 98.82%         | 98.04%              | 97.42%        |

---

## Reflections
- Learned complexities of model selection, training, and evaluation.
- Emphasized understanding performance factors over relying on advanced models.
- Highlighted critical role of data augmentation and hyperparameter tuning.
- Identified minimal impact of certain techniques like shear augmentation.
- Addressed practical challenges: computational limits and data constraints.

---

## Simple API for Model Predictions
### Overview
- Created a Python Flask-based API to provide predictions for insect classification using both custom CNN and pretrained models.
- Hosted in the `Api_Dissertation` folder.

### Folder Structure
- `DenseNet121.keras`, `InceptionV3.keras`, `ResNet50.keras`, `MobileNetV2.keras`, `EfficientNetB0.keras`, `Moderate_augmentation.keras`: Pretrained and custom CNN models.
- `templates/` folder containing the `index.html` file for the API interface.
- `App2.py`: The main Flask application for running the API.

### Features
1. **Pretrained Models**:
   - DenseNet121
   - InceptionV3
   - ResNet50
   - MobileNetV2
   - EfficientNetB0
   - Custom CNN Model

2. **Key Functionality**:
   - Accepts uploaded image files.
   - Provides prediction for selected model.
   - Accessible through a web interface or API endpoint.

3. **Usage Instructions**:
   - Run `App2.py` to start the API.
   - Access the web interface at `http://127.0.0.1:5000/`.
   - Upload an image and select a model for predictions.

### Example Prediction Flow
- **Upload an Image**: Users upload insect images via the web interface.
- **Select a Model**: Choose from DenseNet, Inception, ResNet, MobileNet, EfficientNet, or Custom CNN.
- **Receive Predictions**: The predicted class of the insect is displayed.

---

## Conclusion
This project combined advanced deep learning techniques, robust dataset preparation, and an intuitive API to address the challenges in multi-class insect classification, with a specific focus on the Asian hornet. The implementation provides significant contributions to ecological research and awareness while showcasing the potential of machine learning in biodiversity conservation.
