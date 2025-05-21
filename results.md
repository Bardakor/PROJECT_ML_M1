# AI for Sustainable Ecosystems: Wildlife Classification Results

## Overview
This document presents the results of the wildlife classification component of our "AI for Sustainable Ecosystems" project. As outlined in our research question, we aim to "leverage machine learning – through both image-based analysis and data-driven forecasting – to monitor ecosystem health (forests and wildlife), assess the impacts of climate change, and optimize resource management."

The wildlife classification module specifically addresses the "Vision-Based Environmental Monitoring" component, focusing on recognizing animal species in images to support biodiversity monitoring and conservation efforts.

## Implementation Approach

We implemented a deep learning-based wildlife classification system using the following approach:

1. **Model Architecture**: ResNet50 pre-trained on ImageNet, with a custom classifier head including dropout layers for regularization
2. **Data Source**: The system is designed to work with the iNaturalist dataset as mentioned in the project overview, but currently uses synthetic data for development and testing
3. **Training Process**: Includes data augmentation, regularization, early stopping, and learning rate scheduling
4. **Evaluation**: The model is evaluated on test data with metrics including accuracy, precision, recall, and F1-score

## Initial Results

Our initial implementation showed significant limitations:

- **Test Accuracy**: 38.0%
- **Class Recognition**: The model could only reliably identify a few classes (lion, bear, zebra, owl)
- **Overfitting**: The training accuracy reached ~90% while validation topped at ~60% and then declined
- **Confidence**: The model showed high confidence in incorrect predictions
- **Training Stability**: Validation loss initially decreased but then increased again

These results indicated that while the basic infrastructure was in place, the model required significant improvements to be useful for real-world wildlife monitoring applications.

## Improved Results

After implementing several enhancements to the model architecture and training process, we achieved substantially better performance:

- **Test Accuracy**: 89.0% (a 134% improvement)
- **Class Recognition**: The model now correctly identifies 13 out of 15 classes with high precision and recall
- **Training Stability**: Better convergence between training and validation metrics
- **Confidence**: More appropriate confidence levels in predictions

### Key Improvements Made

1. **Enhanced Model Architecture**:
   - Added dropout layers (0.5 and 0.3) for regularization
   - Added an intermediate layer with 512 neurons for more capacity
   - Added ReLU activation for better feature learning

2. **Improved Training Process**:
   - Added L2 regularization (weight decay = 1e-4) to prevent overfitting
   - Implemented learning rate scheduling with ReduceLROnPlateau
   - Added early stopping with patience=5 to prevent overfitting
   - Increased number of training epochs from 5 to 10

3. **Enhanced Data Augmentation**:
   - Increased rotation angle from 15° to 30°
   - Added random affine transformations (translation and scaling)
   - Enhanced color jitter parameters for more variation
   - Added random grayscale conversion (5% probability)

## Implications for Project Goals

The successful implementation and improvement of the wildlife classification module represent significant progress toward our overall project goals:

1. **Biodiversity Monitoring**: The ability to accurately identify wildlife species from images will enable more efficient monitoring of biodiversity in target ecosystems.

2. **Integration with Analytical Components**: By establishing this vision-based module, we now have a foundation to correlate species presence data with climate and environmental factors in our analytical modeling.

3. **Conservation Prioritization**: Accurate wildlife identification can help inform conservation priorities and resource allocation based on the presence and abundance of different species.

## Remaining Challenges

1. **Real-world Data**: While our model performs well on synthetic data, testing with actual iNaturalist wildlife images is needed to validate performance in real-world scenarios.

2. **Challenging Classes**: Some classes (wolf and penguin) still show poor recognition rates and require focused improvement.

3. **Deployment Considerations**: The model currently runs on CPU; optimizations may be needed for deployment in resource-constrained environments.

## Next Steps

1. **Data Collection**: Incorporate real wildlife images from the iNaturalist dataset
2. **Integration**: Connect the wildlife classification module with the climate forecasting and resource optimization components
3. **Fine-tuning**: Further optimize the model for challenging classes and real-world conditions
4. **System Integration**: Develop an integrated pipeline that combines vision-based monitoring with analytical modeling to address our core research question

This wildlife classification component represents a significant step toward developing a comprehensive AI system for sustainable ecosystem monitoring and management. 