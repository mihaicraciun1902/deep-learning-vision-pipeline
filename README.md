# Deep Learning Computer Vision Pipeline

This project implements a robust deep learning infrastructure in PyTorch to classify low-resolution images across two distinct datasets: facial expressions and medical lesions (skin moles). 

## Tech Stack
* **Core Framework:** PyTorch, Torchvision
* **Data Handling:** Custom `Dataset` and `DataLoader` classes
* **Experiment Tracking:** TensorBoard
* **Optimization:** Adam, AdamW, `SequentialLR` (Warmup + Cosine Annealing)

## Architecture & Features
* **Custom CNN & MLP:** Built and trained a multi-layer perceptron and a 4-layer Convolutional Neural Network from scratch, utilizing `BatchNorm2d` to accelerate convergence and `Dropout` for regularization.
* **Transfer Learning:** Fine-tuned a pre-trained `ResNet18` backbone, adapting the final fully connected layers to the specific class distributions.
* **Data Augmentation:** Implemented on-the-fly transformations (Random Flips, Rotations, Color Jittering) using `torchvision.transforms` to improve model generalization on small datasets.
* **Class Imbalance Mitigation:** Engineered solutions for heavily skewed datasets by implementing a `WeightedRandomSampler` for batch-level balancing and a custom **Focal Loss** function to penalize confident misclassifications.
* **Curriculum Learning (Bonus):** Developed a dynamic pacing strategy to expose the model to progressively more difficult examples as epochs advance.

## Training Methodology
The training loop was rigorously tracked using **TensorBoard**, logging scalar metrics (Train/Val Loss, Macro-F1) and weight histograms. Extensive ablation studies were conducted to prove the efficacy of Learning Rate Warmups, Normalization layers, and Augmentation techniques.
