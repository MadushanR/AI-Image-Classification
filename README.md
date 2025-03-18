# AI-Image-Classification

## Multi-Class Image Classification with Transfer Learning
This project demonstrates how to perform multi-class image classification on the CIFAR-10 dataset using a pretrained MobileNetV2 model (transfer learning) in TensorFlow. The entire training process is containerized with Docker for easy reproducibility and deployment.

## Project Overview
**Objective**: Classify images from 10 distinct classes in the CIFAR-10 dataset (e.g., airplanes, cars, birds, cats, dogs, etc.) <br/>
**Method**: Utilize a pretrained MobileNetV2 as the base model, freezing its layers and adding custom classification layers on top. <br/>
**Benefit**: Transfer learning significantly reduces training time and improves accuracy compared to training from scratch. <br/>

## Features
**Transfer Learning**: Speeds up training by leveraging pretrained ImageNet weights. <br/> 
**Dockerized**: The entire training pipeline runs in a container, ensuring environment consistency. <br/>
**Flexible** **Hyperparameters**: Environment variables (EPOCHS, BATCH_SIZE) let you easily tweak training settings without editing code. <br/>
**Automatic** **Splits**: The dataset is divided into training, validation, and test sets with minimal manual setup. <br/>
**Model** **Saving**: Saves the trained model for inference or future fine-tuning. <br/>

## Tech Stack
**Language**: Python 3.x <br/>
**Framework**: TensorFlow 2.x (Keras) <br/>
**Containerization**: Docker <br/>
**Dataset**: CIFAR-10 (loaded from tf.keras.datasets) <br/>

## Getting Started
**Clone the Repository:** <br/>
git clone https://github.com/MadushanR/AI-Image-Classification.git <br/>
cd AI-Image-Classification

**(Optional)** **Edit** requirements.txt if you want to pin specific versions or add new dependencies.

**Build the Docker Image:** <br/>
docker build -t cifar10-tf:latest .

**Run the Container:** <br/>
docker run --rm -it cifar10-tf:latest <br/>

The script will start training. By default, it runs for 5 epochs with a batch size of 32. 

**Adjust Hyperparameters (optional):** <br/>
docker run --rm -it \
    -e EPOCHS=10 \
    -e BATCH_SIZE=64 \
    cifar10-tf:latest
    
## Usage
**Training:** The container automatically trains the model and outputs logs (epoch-by-epoch accuracy and loss).
**Saving the Model**: By default, the trained model is saved in /app/saved_model inside the container.
**Persisting Model to Host:** Mount a volume to save it locally:
docker run --rm -it \
    -v $(pwd)/saved_model:/app/saved_model \
    cifar10-tf:latest
After training, check your local saved_model/ folder for the exported model.

## Project Structure
cifar10_transfer_learning/
├── Dockerfile
├── requirements.txt
├── train_cifar10.py
└── README.md
Dockerfile: Defines the container environment (base image, dependencies).
requirements.txt: Lists Python packages (e.g., tensorflow).
train_cifar10.py: Main TensorFlow script for loading CIFAR-10, building the model, and training.
README.md: You’re reading it!

## Future Enhancements
**Fine-Tuning:** Unfreeze more (or all) layers in MobileNetV2 with a lower learning rate for higher accuracy.
**Data Augmentation:** Apply random flips/rotations/brightness changes to improve generalization.
**Deployment**: Serve the model with a REST API (Flask or FastAPI) or TensorFlow Serving in Docker.
**MLOps Integration:** Use CI/CD tools (e.g., GitHub Actions, AWS CodeBuild) to automate testing and deployment.
