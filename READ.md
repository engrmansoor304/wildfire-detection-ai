<div align="center">

# 🔥 Wildfire Detection AI System

[![Hugging Face Space](https://img.shields.io/badge/🤗-Live%20Demo-yellow?style=for-the-badge)](https://huggingface.co/spaces/Mansoorrr/wildfire-detection-ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

### 🚀 AI-Powered Satellite Image Analysis for Early Wildfire Detection

[Live Demo](https://huggingface.co/spaces/Mansoorrr/wildfire-detection-ai) • [Report Issue](https://github.com/Mansoorrr/wildfire-detection-ai/issues)

</div>

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [✨ Features](#-features)
- [🏆 Model Performance](#-model-performance)
- [🛠️ Tech Stack](#️-tech-stack)
- [📊 Results](#-results)
- [🚀 Live Demo](#-live-demo)
- [💻 Local Installation](#-local-installation)
- [📁 Project Structure](#-project-structure)
- [🤝 Contributing](#-contributing)
- [📝 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

**Wildfire Detection AI** is an advanced deep learning system that analyzes satellite imagery to detect wildfires in real-time. Using an ensemble of three powerful neural networks, the system achieves **97.22% accuracy** on a dataset of over 21,000 satellite images.

This tool can help:
- 🔥 Early wildfire detection
- 🚒 Rapid emergency response
- 🌍 Environmental monitoring
- 📊 Disaster management planning

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **Ensemble Models** | 3 models working together for maximum accuracy |
| 🔍 **Real-time Detection** | Instant analysis of uploaded satellite images |
| 📊 **Grad-CAM Visualization** | Heatmap explanations of AI decisions |
| 📄 **AI-Powered PDF Reports** | 8-page professional reports via Groq LLaMA3-70B |
| 📈 **Interactive Charts** | Confidence scores & probability breakdowns |
| 📁 **Dataset Explorer** | Browse training data statistics |
| 🎯 **Model Metrics** | Performance comparison & ROC curves |
| 🌙 **Dark Theme UI** | Modern, professional interface |

---

## 🏆 Model Performance

| Model | Architecture | Accuracy | Best For |
|-------|--------------|----------|----------|
| **MobileNetV3** | CNN-based | **97.22%** | 🏆 Best Overall |
| **EfficientNet-B0** | Compound Scaling | **97.03%** | 🎯 High Precision |
| **MobileNetV2** | Depth-wise Conv | **95.76%** | ⚡ Fast Inference |
| **Ensemble** | Majority Voting | **97.22%** | 🔥 Production Ready |

### 📊 Key Metrics
- **AUC Score:** 0.995
- **Test Accuracy:** 96.58%
- **Training Images:** 21,067
- **Inference Speed:** <1 second

---

## 🛠️ Tech Stack

### Deep Learning
| Tool | Purpose |
|------|---------|
| PyTorch | Deep learning framework |
| MobileNetV2/V3 | Lightweight CNN architectures |
| EfficientNet-B0 | High-performance CNN |
| Albumentations | Image augmentation |

### Frontend & Visualization
| Tool | Purpose |
|------|---------|
| Streamlit | Web application framework |
| Plotly | Interactive charts |
| OpenCV | Image processing & Grad-CAM |
| Folium | Map visualization |

### Deployment & LLM
| Tool | Purpose |
|------|---------|
| Hugging Face Spaces | Cloud deployment |
| Docker | Containerization |
| Groq LLaMA3-70B | AI report generation |

---

## 📊 Results

### Detection Examples

| Input | Prediction | Confidence | Grad-CAM |
|-------|------------|------------|----------|
| 🔥 Fire image | WILDFIRE | 94-97% | 🔴 High activation |
| 🌳 Forest image | NO WILDFIRE | 88-95% | 🟢 Low activation |

### Performance Metrics
Classification Report:
precision recall f1-score support
nowildfire 0.97 0.96 0.96 1580
wildfire 0.96 0.97 0.96 1581

accuracy 0.97 3161
macro avg 0.97 0.97 0.97 3161
weighted avg 0.97 0.97 0.97 3161

---

## 🚀 Live Demo

**Try the live application here:**

👉 **[https://huggingface.co/spaces/Mansoorrr/wildfire-detection-ai](https://huggingface.co/spaces/Mansoorrr/wildfire-detection-ai)**

### Demo Instructions:
1. Click the link above
2. Upload a satellite image (JPG, PNG, JPEG)
3. Wait for AI analysis (<1 second)
4. View predictions from all 3 models
5. Explore Grad-CAM heatmaps
6. Download PDF report (optional)

---

## 💻 Local Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)

### Install Dependencies

```bash
pip install -r requirements.txt
MIT License

Copyright (c) 2026 Mansoorrr

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
🙏 Acknowledgments
Groq for free API access to LLaMA3-70B

Hugging Face for free Spaces hosting

PyTorch Team for deep learning framework

Streamlit Team for web app framework

📞 Contact
Mansoor
Project Link: https://github.com/Mansoorrr/wildfire-detection-ai
