# Deep3D-pytorch : A Baseline Implementation of Deep 3D paper in PyTorch

This repository contains a PyTorch implementation of the [Deep 3D paper](https://arxiv.org/abs/1604.03650), a popular method for monocular-to-stereo image conversion. The paper introduces a novel approach for generating depth information from single images to create stereo pairs, leveraging deep learning techniques.

## Purpose of This Implementation
- **Baseline Measurement**: The primary objective of this implementation is to serve as a baseline for evaluating the performance of the Deep 3D method on higher-resolution frames such as 2K and 4K. 
- **Output Analysis**: Assess the quality, performance, and feasibility of this approach for high-resolution images.

## Features
- All you a need is a video to start. Contains ffmpeg modules to generate frames from video.
- Complete reproduction of the Deep 3D methodology in PyTorch.
- Focused on adapting the model to support high-resolution input frames.
- Includes preprocessing, training, and inference scripts for testing on custom datasets.

## Getting Started
1. Clone the repository:
   ```bash
   git clone <repository-url>

