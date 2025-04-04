# Training and Testing the Traffic Sign Recognition Model
This project implements a traffic sign recognition system using a convolutional neural network (CNN). The system is trained on the GTSRB dataset and allows users to upload images of traffic signs to predict their categories using a graphical user interface (GUI) built with tkinter.

## Features
- Model Training: A CNN is trained on the GTSRB dataset to classify traffic signs into 43 categories.
- Prediction GUI: A user-friendly interface for uploading traffic sign images and viewing predictions with accuracy.
- Parameter Tuning: Various parameters were tested to optimize the model's performance.

## Dataset
The model was trained on the GTSRB dataset, which contains images of 43 different traffic sign categories.

## Initial Results after training
After training the model with the the initial parameters and structure granted by the CS50 neural networks project, i decided to test the model, using some random test images i downloaded online.

Despite the ~96% accuracy the model gave out during training, the results where super awkward (1 out of 10 images tested were on track, 10% accurate)


## Parameter Tuning
After seeing how inaccuate my model was, i tried to optimize the output by varying some parameters

1. Number of Filters
As the Feature extractors in the CNN, 

2. Kernel Size
- Increasing the number of kernel layers from two to three and from 16, through 32 to 64, i think the model has a greater chance of performing better
- Using the kernel regularizer from keras improved the accuracy of the model

3. Dropout Rate
- Averaging the number of disabled neurons during training by setting 0.4 helped a lil bit.
- Disabling more neurons during training led to an accuracy of 50% for my test dataset as it highly prevents overfitting
- 

4. Number of Epochs
Increasing the cycles from 10 to 25 added a little bit of precision to the model's prediction


## Getting Started with the Training / using the model
To train the model, run the following command:

### Prerequisites
- Python
- TensorFlow
- Pillow
- Numpy
- sklearn
- Anaconda

1. Clone the repository in your anaconda prompt
```bash
git clone https://github.com/keyz-dev/traffic_model.git
cd traffic_model
```

2. activate tensorflow environment
```bash
conda activate tf
```

4. install the dependencies and packages
```bash
pip install pillow numpy sklearn
```

5. Train the model
```bash
cd CALEB
python traffic.py path/to/data_directory path/to/model.h5
```

6. Use the model
Still within CALEB directory
```bash
python predict_sign.py path/to/model.h5
```