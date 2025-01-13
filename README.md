# Evaluation of Multi Vs. Single-task Neural Networks for Autonomous Driving Perception

## Abstract

A Multi-Task Neural Network (MTNN) enables a single network to solve multiple tasks simultaneously, offering efficiency
and performance improvements over Single-Task Neural Networks (STNNs). This thesis evaluates MTNNs against STNNs for
four autonomous driving perception tasks: semantic segmentation, lane marking, drivable area detection, and object
detection. Using the Audi Autonomous Driving Dataset (A2D2), we observe that MTNNs are up to 33% faster and achieve
higher accuracy across tasks compared to STNNs.

## Repository Overview

This repository contains the code, data preprocessing steps, and results for the thesis "Evaluation of Multi Vs.
Single-task Neural Networks for Autonomous Driving Perception." The project demonstrates the implementation of both
STNNs and MTNNs for autonomous driving perception tasks.

### Directory Structure
```
├── src/ # Python scripts for the project 
├── requirements.txt # List of Python dependencies 
├── README.md # Project documentation 
└── .gitignore # Files/folders to exclude from the repository
```

### Model Architecture

![Alt text](docs/MTNN_architecture.png "MTNN Architecture")


## Features
 - Implementation of STNNs and MTNNs using PyTorch.
 - Tasks supported:
   - Semantic Segmentation
   - Lane Marking
   - Drivable Area Detection
   - Object Detection
 - Use of the Audi Autonomous Driving Dataset (A2D2).
 - Custom loss weighing and optimization techniques.
 - Training scripts and evaluation metrics for comparison.

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, but recommended)

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/thesis-repo.git
   cd thesis-repo

Install the required dependencies:
 ```bash
 pip install -r requirements.txt
 ``` 



###Key findings from the thesis:
- MTNNs are up to 33% faster in inference compared to STNNs. 
- Improved accuracy:
  - Semantic Segmentation: +1%
  - Lane Marking: +2.5% 
  - Drivable Area Detection: +1.5% 
  - Object Detection: +12%
- Transfer learning with MTNN’s shared encoder improved STNN accuracy by up to 2%.
Contributing

Thank you for your interest in this project! If you find it helpful, please star the repository on GitHub.
