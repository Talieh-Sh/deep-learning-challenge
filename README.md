# deep learning challenge
## Alphabet Soup Charity Success Prediction

## Project Overview
The purpose of this project is to help Alphabet Soup charity to identify applications likely to be successful if funded. It is offering a Neural Network model that uses data from past applications and predicts if the application would be successfull if funded or not.

## Dependencies
- TensorFlow
- Scikit-learn
- Pandas
- keras_tuner


## Data Preprocessing
Target Variable: IS_SUCCESSFUL
Features: All columns except EIN and NAME
Encoding: Categorical variables are encoded into numeric values.
Scaling: Feature variables are scaled.

## Model
Architecture: Sequential model with two hidden layers.
Activation: ReLU for hidden layers, Sigmoid for output layer.
Compilation: Adam optimizer, binary crossentropy loss function.

## Training and Evaluation
Split: The data is split into training and testing sets.
Scaling: Data is normalized using StandardScaler.
Evaluation: Model's performance is evaluated using accuracy.

## Export
The model is saved as an HDF5 file, AlphabetSoupCharity.h5, for future use.
