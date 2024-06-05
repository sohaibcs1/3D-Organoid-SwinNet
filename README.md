<div align="center">

  <h1>3D-Organoid-SwinNet</h1>
  <p>This repository contains the code for 3D-Organoid-SwinNet model for Segmentation of 3D organoids. In this readme, you will find a description of 3D-Organoid-SwinNet.</p>

</div>

<div align="center">
  .
  <a href="https://">![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)</a> <br>
  [Download Pretrained model](https://www.dropbox.com/scl/fi/sbewadh8971ez76eu79p4/best_metric_model.pth?rlkey=so4dxnf3qbnt3fuywlqzw6yej&st=d2fhvrre&dl=1)


</div>

<hr />

## Dataset

The data includes organoid cultures fixed on various days and encompasses a broad mutation landscape of breast cancer cell lines, including MCF10A, MCF7, MDA-MB-231, and MDA-MB-468.

## Model Architecture

The proposed framework includes a Swin Transformer block, 3D convolutional blocks, and Multilayer Perceptron (MLP) blocks. The Swin Transformer block generates features at four different resolution scales. After extracting global features with the Swin Transformer block, the architecture employs five 3D convolutional blocks for local feature extraction. Then, MLP blocks are used to reconstruct the output shape. Finally, an additional 3D convolutional layer with a 1×1 kernel size acts as the model's terminal layer.

![Architecture](https://github.com/sohaibcs1/3D-Organoid-SwinNet/blob/main/images/architecture.JPG)

# Project Tree Structure

- 3D-Organoid-SwinNet  
  - data 
    - images
    - masks
    - dataset.json 
  - images
  - model
  - saved_model
  - train
  - utills
  - README.md
  - requirements.txt
  - train_model.py

## Running the Model
To run the model, follow these steps:

1. **Install Dependencies**: Ensure you have all the required dependencies installed. Navigate to the root directory of the project in your terminal and execute the following command:

    ```
    pip install -r requirements.txt
    ```

    This command will install all the necessary Python packages listed in the `requirements.txt` file.


2. **Execute the Training Script**: Run the training script `train_model.py` to start the training process for model.

    ```
    python train_model.py
    ```

    Make sure to adjust any parameters or configurations in the `train_model.py` script according to your requirements before running it.

Ensure that you have a suitable Python environment set up and configured before proceeding with the steps above.

## Results
The figure compares the results obtained from our approach with those from other techniques, demonstrating that our model, 3D-Organoid-SwinNet, outperforms the alternatives.

![Result](https://github.com/sohaibcs1/3D-Organoid-SwinNet/blob/main/images/results.JPG)
