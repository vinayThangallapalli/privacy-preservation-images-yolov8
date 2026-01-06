# Privacy Preservation for Images for Public Privacy using YOLOv8

This project focuses on preserving public privacy in images by detecting sensitive
objects such as faces and people using YOLOv8 and applying anonymization techniques.
The system ensures that identifiable information is protected before storage or sharing.

## Objectives
- Detect privacy-sensitive regions in images using YOLOv8
- Apply anonymization techniques such as blurring or masking
- Preserve usability of images while protecting identity
- Enable privacy-aware image processing for public environments

## Features
- YOLOv8-based object detection
- Face and person anonymization
- Image preprocessing pipeline
- Modular and scalable design

## Tech Stack
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- Streamlit / FastAPI (optional UI)

## Project Structure
Src/                # Source code
Input/              # Input videos
encrypted_frames/   # Encrypted frames
metadata/           # Metadata files
Output/             # Final output
temp/               # Temporary files

## How It Works
1. Input image is provided to the system  
2. YOLOv8 detects privacy-sensitive objects  
3. Detected regions are anonymized  
4. Privacy-preserved image is generated  

## Use Cases
- Public surveillance systems
- Smart cities
- Privacy-compliant image sharing
- Research and academic demonstrations

