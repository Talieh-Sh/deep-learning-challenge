# deep learning challenge
## Alphabet Soup Charity Success Prediction

## Project Overview
This project aims to help Alphabet Soup charity identify applications likely to be successful if funded. It offers a Neural Network model that uses data from past applications and predicts if the application would be successful if funded or not.


## Google Colab
To facilitate the development, training, and evaluation of our neural network model, we utilized Google Colab. 
Google Colab is a free cloud service that supports Python programming and provides a robust environment for machine learning and data analysis projects.

### Advantages of Using Google Colab
- Pre-installed Libraries
- High-Performance Computing: Access to free GPU and TPU resources in Colab.
- No Local Setup Required
- Collaboration

  
## Dependencies
- TensorFlow
- Scikit-learn
- Pandas
- keras_tuner


## Data Preprocessing
- **Target Variable:** IS_SUCCESSFUL
- **Features:** All columns except EIN and NAME (APPLICATION_TYPE,	AFFILIATION,	CLASSIFICATION,	USE_CASE,	ORGANIZATION,	STATUS,	INCOME_AMT,	SPECIAL_CONSIDERATIONS, ASK_AMT)
- **Encoding:** Categorical variables are encoded into numeric values. using: pd.get_dummies

## Model
- **Architecture:** Sequential model with hidden layers.
- **Activation:** ReLU, Sigmoid, ...  for hidden layers and output layer.
- **Compilation:** Adam optimizer, binary crossentropy loss function.

## Training and Evaluation
- **Split:** The data is split into training and testing sets.
- **Evaluation:** The Model's performance is evaluated using accuracy.

## Export
- The model is saved as an HDF5 file, AlphabetSoupCharity.h5, for future use.

# Optimization with Keras Tuner

In the AlphabetSoupCharity_Optimisation model:
We used Keras Tuner library to improve the accuracy of the model.Keras Tuner is used to automate the hyperparameter tuning process.
## Dynamic Architecture with Hyperparameter Tuning: 
The optimized model used Keras Tuner to dynamically determine the best architecture and hyperparameters.
- **Activation Functions:** It allowed for the choice between multiple activation functions (relu, tanh, elu, selu, leakyrelu) for the hidden layers.
- **Neurons in Layers:** The number of neurons in the first layer and subsequent hidden layers was not fixed but chosen from a range (1 to 30, with step increments of 5) by Keras Tuner.
- **Number of Hidden Layers:** Keras Tuner decided on the number of hidden layers (between 1 and 6), allowing the model to explore various depths.
