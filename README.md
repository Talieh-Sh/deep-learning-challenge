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

**Result: loss: 0.8103 - accuracy: 0.6546**
**************************************
# Optimization with Keras Tuner
**************************************
In the AlphabetSoupCharity_Optimisation model (AlphabetSoupCharity_Optimisation.ipynb):
We used the Keras Tuner library to improve the accuracy of the model. Keras Tuner is used to automate the hyperparameter tuning process.
## Dynamic Architecture with Hyperparameter Tuning: 
The optimized model used Keras Tuner to dynamically determine the best architecture and hyperparameters.
- **Activation Functions:** It allowed for the choice between multiple activation functions (relu, tanh, elu, selu) for the hidden layers.
- **Neurons in Layers:** The number of neurons in the first layer and subsequent hidden layers was not fixed but chosen from a range (1 to 30, with step increments of 5) by Keras Tuner.
- **Number of Hidden Layers:** Keras Tuner decided on the number of hidden layers (between 1 and 6), allowing the model to explore various depths.

**Result: Loss: 0.5593386888504028, Accuracy: 0.7271137237548828** 

**************************************
# Changing cutoff limits
**************************************
AlphabetSoupCharity_Optimisation_2.ipynb 
Changing the cutoff limit for Application Type from 500 to 1000 (from 9 bins to 6 bins) 
Changing the cutoff limit for Classification Type from 1000 to 2000 (from 6 bins to 4 bins) 

**Result: Loss: 0.5639263987541199, Accuracy: 0.7248979806900024** 



**************************************
# Using Dropout and L2 Regularization
**************************************
incorporating comprehensive hyperparameter tuning that includes regularization parameters and employing strategies to combat overfitting, such as dropout and L2 regularization. The expanded search space for hyperparameters and the inclusion of early stopping suggest a model designed for enhanced performance and generalization. 


**Result: Test Loss: 0.5692663192749023, Test Accuracy: 0.7203498482704163**
**************************************
# An overview of important concepts in NN models:
**************************************
- Epoch: An epoch represents one complete pass of the training dataset through the neural network, involving both a forward pass and a backward pass for all training examples.

- Batch Size: The number of training examples utilized in one iteration of model training. The entire dataset is divided into numerous batches when training a model.

- Neurons: Basic units of computation in a neural network, neurons receive inputs, process them (often with a weighted sum followed by a non-linear activation), and produce an output.

- Layers: Collections of neurons organized in a network. Layers are structured into input, hidden, and output layers, with each serving different roles in processing data and making predictions.

- Activation Function: A mathematical function applied to the neurons' output, introducing non-linearity into the model, enabling it to learn complex patterns. Examples include ReLU, sigmoid, and tanh.

- Learning Rate: A hyperparameter that controls how much the model's weights are updated during training. A smaller learning rate requires more training epochs, while a larger learning rate may lead to rapid convergence but can overshoot the minimum loss.

- Optimizer: An algorithm or method used to change the attributes of the neural network, such as weights and learning rate, to reduce losses. Optimizers include SGD (Stochastic Gradient Descent), Adam, and RMSprop.

- Loss Function: A method to calculate the difference between the model's predictions and the actual data. It guides the optimizer by indicating how well the model is performing. Common loss functions include binary crossentropy for binary classification tasks.

- Validation Data: A subset of the dataset not used in training. It's used to evaluate the model during training, providing feedback on how well the model generalizes to unseen data.

- Hyperparameters: Parameters that are set prior to the commencement of the learning process, guiding the training algorithm. These include learning rate, batch size, and architecture choices like the number of layers and neurons per layer.

- Keras Tuner: A library for TensorFlow to automate the process of selecting the best hyperparameters for your model, enhancing model performance by finding the optimal configuration.
  
