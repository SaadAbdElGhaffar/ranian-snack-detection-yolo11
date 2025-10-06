# 🧃 Iranian Snack Detection with YOLOv11 📊

A comprehensive computer vision project for detecting and analyzing Iranian supermarket snacks and chips using YOLOv11 object detection and segmentation models. This system combines real-time object detection with product metadata analysis to provide valuable insights for retail environments.

![Project Banner](https://img.shields.io/badge/AI-YOLOv11-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Project Overview

This project focuses on detecting and analyzing popular Iranian supermarket snacks and chips using advanced computer vision techniques. The system can:

- **Detect & Segment** 19 different Iranian snack products
- **Calculate Total Price** of detected items
- **Monitor Expiry Dates** and warn about products nearing expiration
- **Calculate Total Calories** for nutritional analysis
- **Process Videos** with real-time annotations
- **Generate Analytics** for retail inventory management

## Demo

Watch the system in action! The GIF below shows real-time detection and analysis of Iranian snack products with price, expiry date, and calorie information overlayed on the video:

![Iranian Snack Detection Demo](outputs/iranian_snack_detection_annotated_video.gif)

*Real-time detection showing total price, nearest expiry date, and total calories for detected products*

## �🏷️ Detected Product Categories

The model is trained to detect 19 popular Iranian snack products:

- **Ashi Mashi Snacks**
- **Chee Pellet** (Ketchup & Vinegar flavors)
- **Cheetoz** (Multiple varieties: Chili, Ketchup, Onion & Parsley, Salty, Vinegar, Wheel snacks)
- **Maz Maz** (Ketchup chips, Potato sticks, Salty & Vinegar chips)
- **Mini Lina**
- **Minoo Cream Biscuit**
- **Naderi** (Mini Cookie & Mini Wafer)

## 📁 Project Structure

```
iranian-snack-detection/
├── 📂 data/                              # Dataset and product information
│   ├── Products_info(1404-03-14).csv    # Product metadata (prices, calories, etc.)
│   ├── sample_video_20250506_213503.mp4 # Sample video for testing
│   └── Iranian Snack and Chips Detection (YOLO Format)/
│       ├── data.yaml                     # YOLO dataset configuration
│       ├── train/                        # Training images and labels
│       ├── valid/                        # Validation images and labels
│       └── test/                         # Test images and labels
├── 📂 models/                            # Pre-trained model weights
│   ├── yolo11m-seg.pt                   # YOLOv11 Medium Segmentation model
│   └── yolo11n.pt                       # YOLOv11 Nano model
├── 📂 notebooks/                        # Jupyter notebooks
│   ├── iranian-snack-detection-yolo.ipynb  # Main analysis notebook
│   └── runs/                            # Training outputs and predictions
├── 📂 outputs/                          # Generated videos and results
├── 📋 requirements.txt                  # Python dependencies
├── 🚫 .gitignore                       # Git ignore rules
└── 📖 README.md                        # This file
```

## ⚠️ Important Notes

### 📅 Project Timeline & Date Sensitivity
- **Development Period**: This project was developed and tested over 3 months (July 2025 - October 2025)
- **Upload Date**: First uploaded to GitHub on October 6, 2025
- **Expiry Date Warning**: If you run the notebook today, the calculated expiry dates will differ from the demo results since the product data uses specific Persian calendar dates from the original development period
- **Date Dependencies**: The system calculates "days to expiry" based on the current date, so results will vary depending on when you run the analysis

### 💾 File Availability & GitHub Limitations
- **Original Platform**: Developed on Kaggle
- **GitHub Constraints**: The complete runs folder is too large for GitHub upload, so we only kept essential files:
  - ✅ Best model weights (needed for inference)
  - ✅ Key training results (performance metrics)
  - ✅ Sample outputs and predictions  
  - ❌ Complete training logs and checkpoints (too large for GitHub)
- **Note**: For full training reproduction, use Kaggle or Google Colab

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster training/inference)
- FFmpeg (for video processing)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/iranian-snack-detection.git
   cd iranian-snack-detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download pre-trained models:**
   - The YOLOv11 models will be automatically downloaded when first used
   - Or manually place your trained models in the `models/` folder

### Usage

#### 🔍 Image Detection

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('models/yolo11m-seg.pt')

# Run prediction on images
results = model.predict(
    source='path/to/your/images',
    save=True,
    conf=0.8
)
```

#### 🎥 Video Processing

```python
# Process video with annotations
results = model.predict(
    source='path/to/video.mp4',
    save=True,
    save_txt=True,
    conf=0.8
)
```

#### 📊 Run Complete Analysis

Open and run the Jupyter notebook:

```bash
jupyter notebook notebooks/iranian-snack-detection-yolo-eda.ipynb
```

## 📈 Features & Capabilities

### 🎯 Object Detection & Segmentation
- **High Accuracy**: 99%+ precision and recall on validation set
- **Instance Segmentation**: Precise pixel-level masks for each product
- **Multi-class Detection**: Simultaneous detection of multiple product types

### 💰 Business Intelligence
- **Price Calculation**: Automatic total price computation
- **Expiry Monitoring**: Days-to-expiry tracking and warnings
- **Calorie Analysis**: Nutritional information aggregation
- **Inventory Insights**: Data-driven product analysis

### 🎬 Video Analysis
- **Real-time Processing**: Frame-by-frame analysis and annotation
- **Dynamic Overlays**: Live price, expiry, and calorie information
- **Export Capabilities**: Annotated video generation

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **mAP50 (Box)** | 99.1% |
| **mAP50-95 (Box)** | 95.8% |
| **mAP50 (Mask)** | 98.7% |
| **mAP50-95 (Mask)** | 94.2% |
| **Precision** | 99.3% |
| **Recall** | 98.9% |

## 🛠️ Training Your Own Model

1. **Prepare your dataset** in YOLO format
2. **Update data.yaml** with your class names and paths
3. **Run training:**
   ```python
   from ultralytics import YOLO
   
   model = YOLO('yolo11m-seg.pt')
   results = model.train(
       data='data/your_dataset/data.yaml',
       epochs=50,
       batch=16,
       imgsz=640,
       name='your_model_name'
   )
   ```

## 📋 Dataset Information

The dataset includes:
- **Training Set**: 452 images with annotations
- **Validation Set**: 91 images with annotations  
- **Test Set**: 60 images with annotations
- **Product Metadata**: CSV file with pricing, nutritional, and expiry information

### Data Format
- **Images**: JPG format, various resolutions
- **Annotations**: YOLO format (normalized coordinates)
- **Metadata**: CSV with Persian/Jalali date support

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics** for the amazing YOLOv11 framework
- **Roboflow** for dataset management and annotation tools
- **OpenCV** and **PIL** for image processing capabilities
- The open-source community for continuous inspiration

---
⭐ **If you find this project helpful, please give it a star!** ⭐