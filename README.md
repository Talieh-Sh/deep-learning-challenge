# deep learning challenge
## Alphabet Soup Charity Success Prediction

## Project Overview
This project aims to help Alphabet Soup charity identify applications likely to be successful if funded. It offers a Neural Network model that uses data from past applications and predicts if the application would be successful if funded or not.

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
Split: The data is split into training and testing sets.
Scaling: Data is normalized using StandardScaler.
Evaluation: Model's performance is evaluated using accuracy.

## Export
The model is saved as an HDF5 file, AlphabetSoupCharity.h5, for future use.
