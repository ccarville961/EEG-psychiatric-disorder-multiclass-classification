# Imports for the project
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras import activations
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.metrics import RootMeanSquaredError
import matplotlib.pyplot as plt

# Data being assigned from csv to variable.
brainStateData = pd.read_csv('assignment2Data/brainStateData.csv')
# Printing out first 5 rows of data to observe raw data.
print('\n----------------------------------------------------\n')
print(brainStateData.head())

brainStateData.info()

# Plot used to count how many units of the data applied to each label.
sns.countplot(x='label', data=brainStateData)

# Setting variables with the extracted data to be used in visualisations to understand the data better.
brainStateSchizophreniaData = brainStateData.loc[brainStateData["label"] == "SCHIZOPHRENIA"]
# The data columns selected represented the EEG signals recorded and not the mean / average signals
# providing a more realistic visual.
brainStateExtractedSchizophreniaData = brainStateSchizophreniaData.loc[2, 'fft_0_b':'fft_749_b']

# Plotting extracted schizophrenia data.
# The plot shows the signals for schizophrenia typically range in between -600 to 600
plt.figure(figsize=(15, 10))
plt.plot(range(len(brainStateExtractedSchizophreniaData)), brainStateExtractedSchizophreniaData)
plt.title("Extracted schizophrenia data")

# Setting variables with the extracted data to be used in visualisations to understand the data better.
brainStateDepressionData = brainStateData.loc[brainStateData["label"] == "DEPRESSION"]
# The data columns selected represented the EEG signals recorded and not the mean / average signals
# providing a more realistic visual.
brainStateExtractedDepressionData = brainStateDepressionData.loc[0, 'fft_0_b':'fft_749_b']

# Plotting extracted depression data.
# The plot shows the signals for depression typically range in between -600 to 600
plt.figure(figsize=(15, 10))
plt.plot(range(len(brainStateExtractedDepressionData)), brainStateExtractedDepressionData)
plt.title("Extracted depression data")

# Setting variables with the extracted data to be used in visualisations to understand the data better.
brainStateHealthyData = brainStateData.loc[brainStateData["label"] == "HEALTHY"]
# The data columns selected represented the EEG signals recorded and not the mean / average signals
# providing a more realistic visual.
brainStateExtractedHealthyData = brainStateHealthyData.loc[1, 'fft_0_b':'fft_749_b']

# Plotting extracted healthy data.
# The plot shows the signals for healthy typically range in between -50 to 250
plt.figure(figsize=(15, 10))
plt.plot(range(len(brainStateExtractedHealthyData)), brainStateExtractedHealthyData)
plt.title("Extracted healthy data")

# Checks data for missing values and spits a sum of entries missing
print('\n----------------------------------------------------\n')
print('Number of missing entries: ', brainStateData.isnull().sum())

# This variable assigns numerical values to replace the string data labels so that the labels can be used as part of
# the dataframe
encodeCategoricalValues = ({'HEALTHY': 0, 'SCHIZOPHRENIA': 1, 'DEPRESSION': 2})
# New dataframe with encoded values assigned
brainStateDataEncoded = brainStateData.replace(encodeCategoricalValues)
# Printing out first 5 rows of data to observe raw data
print('\n----------------------------------------------------\n')
print(brainStateDataEncoded.head())
# Printing out the count for how many labels appear as part of the data
print('\n----------------------------------------------------\n')
print(brainStateDataEncoded['label'].value_counts())
# Finds the unique elements of an array and returns these unique elements as a sorted array
brainStateDataEncoded['label'].unique()
# Printing out first 5 rows of data to observe raw data with the encoded column
brainStateDataEncoded.head()
# Variable to contain all the eeg data without the label data column
brainStateDataWithoutLabelColumn = brainStateDataEncoded.drop(["label"], axis=1)
print('\n----------------------------------------------------\n')
print('X Shape: ', brainStateDataWithoutLabelColumn.shape)
# Variable to contain the label data column
brainStateDataWithOnlyLabelColumn = brainStateDataEncoded.loc[:, 'label'].values
print('\n----------------------------------------------------\n')
print('Y Shape: ', brainStateDataWithOnlyLabelColumn.shape)

# LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# Need this aswell for after the prediction so that we can inverse scale

# Variable removes the means and scales each feature/variable to unit variance
scaler = StandardScaler()
# Method helps in fitting the data into a model
scaler.fit(brainStateDataWithoutLabelColumn)
# The transform method helps to transform the data so that it is suitable for a model
brainStateDataWithoutLabelColumn = scaler.transform(brainStateDataWithoutLabelColumn)
# Converts a class vector (integers) to binary class matrix. E.g. for use with categorical_crossentropy
brainStateDataWithOnlyLabelColumn = to_categorical(brainStateDataWithOnlyLabelColumn)

# The method assigns the data for the model to be tested against and in this case for the x axis the data without the
# label column has been assign and for the y axis the data containing the labels has been assigned. The test size has
# been set to 0.2 meaning 20% of the data will be kept for testing the model vs the other 80% used for training.
# Random state is set to 4 keeping the seed set the same. This means the testing and training data will remain the
# same regardless of how many times the code is executed.
# Training data accuracy score:  0.9695550351288056 on average the most stable score.
xTrain, xTest, yTrain, yTest = train_test_split(brainStateDataWithoutLabelColumn, brainStateDataWithOnlyLabelColumn,
                                                test_size=0.4, random_state=4)

# Training data accuracy score:  0.9640625 accuracy drops by 1-2% on average with less training data.
# xTrain, xTest, yTrain, yTest = train_test_split(brainStateDataWithoutLabelColumn, brainStateDataWithOnlyLabelColumn,
#                                                 test_size=0.3, random_state=4)

# Training data accuracy score:  0.9636576787807737 drops by 1-2% on average with less training data.
# xTrain, xTest, yTrain, yTest = train_test_split(brainStateDataWithoutLabelColumn, brainStateDataWithOnlyLabelColumn,
#                                                 test_size=0.4, random_state=4)

# This method takes in an array to be reshaped in this case it is the x axis training data, the shape from the first
# column, int value 1 as with multiplication times * 1 will not affect result and final variable gives the number of
# columns.
xTrain = np.reshape(xTrain, (xTrain.shape[0], 1, brainStateDataWithoutLabelColumn.shape[1]))
xTest = np.reshape(xTest, (xTest.shape[0], 1, brainStateDataWithoutLabelColumn.shape[1]))

# Printing out the shapes of the testing and training data to further understand the split of the data.
print('\n----------------------------------------------------\n')
print('X Train Shape: ', xTrain.shape)
print('\n----------------------------------------------------\n')
print('X Test Shape: ', xTest.shape)
print('\n----------------------------------------------------\n')
print('Y Train Shape: ', yTrain.shape)
print('\n----------------------------------------------------\n')
print('Y Test Shape: ', yTest.shape)
print('\n----------------------------------------------------\n')

# The method below from the keras package resets all states generated by keras.
tf.keras.backend.clear_session()

# Creating a variable to contain the sequential model that will hold a plain stack of layers
lstmModel = Sequential()

# Training data accuracy score:  0.9695550351288056
# Adding the LSTM layer that returns a sequence of vectors with 100 dimensions.
# Set the input layer to a 1*2548 matrix.
# Activation for this layer set to relu meaning neurons that have a linear output less than 0 get deactivated,
# this also improves computational speeds with less neurons being activated.
# Return sequence set to true to return a sequence for next lstm process.
lstmModel.add(LSTM(100, input_shape=(1, 2548), activation="relu", return_sequences=True))
# This dropout layer randomly assigns 0 to 20% of the available units to prevent against overfitting
# and units not dropped are scaled to a 1 to 1 rate so that the overall input is unchanged.
lstmModel.add(Dropout(0.2))
# Adding the LSTM layer that returns a sequence of vectors with 50 dimensions.
# Activation for this layer set to sigmoid meaning neurons will be used to predict the probability of the
# output which in this case is very useful for categorising the model.
# Return sequence set to false by defaults as next layer is dropout layer
# (this returns a sequence/vector that goes into your dense layer).
lstmModel.add(LSTM(50, activation="sigmoid"))
# This dropout layer randomly assigns 0 to 20% of the available units to prevent against overfitting
# and units not dropped are scaled to a 1 to 1 rate so that the overall input is unchanged.
lstmModel.add(Dropout(0.2))

# Training data accuracy score:  0.9531615925058547 Dropped by 2-3% when using two relus
# lstmModel.add(LSTM(100, input_shape=(1, 2548), activation="relu", return_sequences=True))
# lstmModel.add(Dropout(0.2))
# lstmModel.add(LSTM(50, activation="relu"))
# lstmModel.add(Dropout(0.2))

# Training data accuracy score:  0.9601873536299765 Dropped by 2-3% when using 1 relu and 1 swish
# lstmModel.add(LSTM(100, input_shape=(1, 2548), activation="relu", return_sequences=True))
# lstmModel.add(Dropout(0.2))
# lstmModel.add(LSTM(50, activation="relu"))
# lstmModel.add(Dropout(0.2))

# Training data accuracy score:  0.9625292740046838 Dropped by 2-3% when using 1 relu and 1 selu
# lstmModel.add(LSTM(100, input_shape=(1, 2548), activation="relu", return_sequences=True))
# lstmModel.add(Dropout(0.2))
# lstmModel.add(LSTM(50, activation="selu"))
# lstmModel.add(Dropout(0.2))

# Training data accuracy score:  0.9765807962529274 Dropped by 2-3% when using two selus
# lstmModel.add(LSTM(100, input_shape=(1, 2548), activation="relu", return_sequences=True))
# lstmModel.add(Dropout(0.2))
# lstmModel.add(LSTM(50, activation="relu"))
# lstmModel.add(Dropout(0.2))

# Training data accuracy score:  0.9461358313817331 Dropped by 2-3% 1 relu and 1 softmax
# lstmModel.add(LSTM(100, input_shape=(1, 2548), activation="relu", return_sequences=True))
# lstmModel.add(Dropout(0.2))
# lstmModel.add(LSTM(50, activation="relu"))
# lstmModel.add(Dropout(0.2))

# Below is an example of hyper tuning the units within the LSTM layers / cells, with the units set to a much lower
# quantity the model was subject to underfitting and this can be seen in the plots in the write up.
# When adding more units per layer you are essentially adding more nodes into the hidden layer which allows the model to
# add a wide variety of implicit relationships among the inputs received.
# When adding more layers to the model it will hold both relationships among inputs and derived data from the sequence
# which increased the depth of the relationships of the model.
# lstmModel.add(LSTM(8, input_shape=(1, 2548), activation="relu", return_sequences=True))
# lstmModel.add(Dropout(0.2))
# lstmModel.add(LSTM(4, activation="sigmoid"))
# lstmModel.add(Dropout(0.2))

# The dense layer transforms the model to a 3 dimensional space and with the sigmoid function applied
# the categorical output will processed with a defined probability score as the data is non linear but categorical.
lstmModel.add(Dense(3, activation='sigmoid'))

# The below code was used to check model call backs. The parameter save best only set to true tells the code to save
# only the best model which is the one with the lowest validation loss which is important for the model predicting
# with high level accuracy on new unseen data.
lstmModelCheckPoint = tf.keras.callbacks.ModelCheckpoint('lstmModel', save_best_only=True)

# Model compilation is an activity performed after writing the statements in a model and before training starts.
# It checks for format errors, and defines the loss function, the optimizer or learning rate, and the metrics.
# A compiled model is needed for training but not necessary for predicting.
# The .compile method utilises the loss parameter 'categorical crossentropy' as I have 3 potential outputs for
# my model prediction (healthy, schizophrenia, depression). The output is one hot encoded in form 1s and 0s.
# Adam realizes the benefits of both AdaGrad and RMSProp. Adam also makes use of the average of the second
# moments of the gradients (the uncentered variance). Specifically, the algorithm calculates an exponential
# moving average of the gradient and the squared gradient, and the parameters beta1 and beta2 control the
# decay rates of these moving averages. The initial value of the moving averages and beta1 and beta2 values
# close to 1.0 (recommended) result in a bias of moment estimates towards zero. This bias is overcome by first
# calculating the biased estimates before then calculating bias-corrected estimates.
# By default the learning rate in the adam optimiser is set to a minimal value 0.00001 (the higher this value
# is the faster the model attempts to decrease the loss, the downside of this is the local minimums may be missed
# from the data making the model less accurate overall)
# The metrics parameter measures the total and count of which the predicted and true y axis values match.

# Training data accuracy score:  0.9695550351288056
# Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order
# and second-order moments.
# According to Kingma et al., 2014, the method is "computationally efficient, has little memory requirement,
# invariant to diagonal rescaling of gradients, and is well suited for problems that are large in terms of
# data/parameters".
lstmModel.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
# Training data accuracy score:  0.9695550351288056
# lstmModel.compile(optimizer="adam", loss=MeanSquaredError(),  metrics=[RootMeanSquaredError()])

# Training data accuracy score:  0.9765807962529274 very inconsistent ranged differences of 3-5%
# Nadam Optimization Algorithm. The Nesterov-accelerated Adaptive Moment Estimation, or the Nadam, algorithm
# is an extension to the Adaptive Movement Estimation (Adam) optimization algorithm to add Nesterov's Accelerated
# Gradient (NAG) or Nesterov momentum, which is an improved type of momentum.
# lstmModel.compile(optimizer="Nadam", loss='categorical_crossentropy',  metrics=['accuracy'])

# Training data accuracy score:  0.7775175644028103
# SGD algorithm is an extension of the Gradient Descent and it overcomes some of the disadvantages of
# the GD algorithm. Gradient Descent has a disadvantage that it requires a lot of memory to load the
# entire dataset of n-points at a time to compute the derivative of the loss function.
# lstmModel.compile(optimizer="sgd", loss=MeanSquaredError(),  metrics=[RootMeanSquaredError()])

# Training data accuracy score:  0.3091334894613583
# Follow The Regularized Leader" (FTRL) is an optimization algorithm developed at Google for
# click-through rate prediction in the early 2010s. It is most suitable for shallow models with
# large and sparse feature spaces.
# lstmModel.compile(optimizer="ftrl", loss=MeanSquaredError(),  metrics=[RootMeanSquaredError()])

# Training data accuracy score:  0.9672131147540983
# The RMSprop optimizer is similar to the gradient descent algorithm with momentum.
# The RMSprop optimizer restricts the oscillations in the vertical direction.
# Therefore, we can increase our learning rate and our algorithm could take larger
# steps in the horizontal direction converging faster.
# lstmModel.compile(optimizer="RMSprop", loss='categorical_crossentropy',  metrics=['accuracy'])

# Prints the information about the model, the shape of the model and available parameters of the model and
# highlights how many of the parameters were dropped during the dropout layers of the model.
lstmModel.summary()


# The .fit method utilised 4 parameters, two for the x and y training data, the validation data to test the
# accuracy of the predictions from the training by measuring the loss and any model metrics.
# The number of epochs to train the model is set to 100. An epoch is an iteration over the entire x and y data provided.
# The callbacks parameter checks after every epoch if the model is to be saved depending on if the validation loss is
# higher or lower than it was before and it will save it based on the absolute minimum calculated.
# When using these alternative settings the accuracy of the model decreased as with less epochs resulted in
# less runs where the weights are altered in the NN and the boundary goes from underfitting to optimal to overfitting.
# Underfitting is bad because the accuracy / performance of the model degrades meaning the model will not provide
# accurate results on unseen data.
# Overfitting is bad because if the model overfits to its training data it will not be able to generalise to new unseen
# data rendering the model useless when deployed in the professional environment as it will be only fit to handle the
# data it is trained on.
# The difference on average in performance was 2%, this may seem like a small amount but when handling data with
# hundreds of millions of data points this can have a land slide effect and cause complete inaccuracies when the
# data is scaled to a big data level.
eegFittedLstmModel = lstmModel.fit(xTrain, yTrain, epochs=100, validation_data=(xTest, yTest),
                                   callbacks=[lstmModelCheckPoint])
# Returns the loss value & metrics values for the model in test mode.
# X represents input and y target data for evaluation.
print('\n----------------------------------------------------\n')
print(lstmModel.evaluate(xTest, yTest))

# Predict method generates output predictions for the input xTest data.
# Computation is done in batches. This method is designed for batch processing of large numbers of inputs.
lstmModelPrediction = lstmModel.predict(xTest)
# Using numpy argmax to return the indices of the maximum values along the axis 1 represents horizontal
predictedLabelsForEEGData = np.argmax(lstmModelPrediction, axis=1)
actualLabelsForEEGData = np.argmax(yTest, axis=1)
# Printing the true labels and the predicted labels to showcase differences in results
print('\n----------------------------------------------------\n')
print(actualLabelsForEEGData)
print('\n----------------------------------------------------\n')
print(predictedLabelsForEEGData)
# printing the results of the true vs predicted labels to determine how accurate the model was.
print('\n----------------------------------------------------\n')
print("Training data accuracy score: ", accuracy_score(actualLabelsForEEGData, predictedLabelsForEEGData))

# Below the method printed out the classification report for the models prediction accuracy of the x axis data.
print('\n----------------------------------------------------\n')
print(classification_report(np.argmax(yTest, axis=1), np.argmax(lstmModel.predict(xTest), axis=1)))

# This plot visually shows a comparison between the training data loss vs the validation loss.
plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.plot(eegFittedLstmModel.history['loss'])
plt.plot(eegFittedLstmModel.history['val_loss'])
plt.title("EEG LSTM Model Loss")
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.legend(['Training data loss', 'Validation / Test data loss'], loc='best')

# This plot visually shows a comparison between the accuracy of the training data vs the accuracy of the validation
# data.
plt.subplot(1, 2, 2)
plt.plot(eegFittedLstmModel.history['accuracy'])
plt.plot(eegFittedLstmModel.history['val_accuracy'])
plt.title("EEG LSTM Model Accuracy")
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.legend(['Training data accuracy', 'Validation / Test data accuracy'], loc='best')
# Method prints out all of the graphs in the python file.
plt.show()

# Sources for content and data
# https://archive.physionet.org/physiobank/database/eegmat/
# https://youtu.be/tepxdcepTbY
# https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions?resource=download
# https://www.kaggle.com/code/utkunoyan/eeg-emotion-lstm/notebook
# https://www.google.com/search?q=python+.unique()&oq=python+.unique()&aqs=chrome..69i57j0i22i30l9.10081j0j7&sourceid=chrome&ie=UTF-8
# https://www.google.com/search?q=.fit%28%29+python&sxsrf=APwXEde9u1ERl2koBNJ87rLgEShRSvSQeA%3A1682542166957&ei=Vo5JZInwOaCkkdUPusydmAM&oq=.fit%28&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQARgAMgoIABCABBAUEIcCMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEOgkIABAHEB4QsAM6CAgAEIAEELADSgQIQRgBUK0VWK0VYL0oaAFwAHgAgAFOiAFOkgEBMZgBAKABAcgBCsABAQ&sclient=gws-wiz-serp
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
# https://towardsdatascience.com/why-do-we-set-a-random-state-in-machine-learning-models-bb2dc68d8431
# https://www.tensorflow.org/api_docs/python/tf/keras/backend/clear_session
# https://keras.io/guides/sequential_model/
# https://gretel.ai/gretel-synthetics-faqs/how-many-epochs-should-i-train-my-model-with#:~:text=The%20right%20number%20of%20epochs,again%20with%20a%20higher%20value.
# https://www.freecodecamp.org/news/improve-image-recognition-model-accuracy-with-these-hacks/#:~:text=Increasing%20epochs%20makes%20sense%20only,with%20your%20model's%20learning%20rate.
# https://www.tutorialspoint.com/keras/keras_models.htm#:~:text=Sequential%20API%20is%20used%20to,and%20output%20to%20the%20model.
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout
# https://www.google.com/search?q=rmsprop+optimizer&sxsrf=APwXEddyXqKP-KcONQHS-Tfp1S-ZmEMVvA%3A1682858858184&ei=amNOZOPxCqOFhbIPlpqTYA&oq=rms+optimizer&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAxgAMgYIABAHEB4yBggAEAcQHjIGCAAQBxAeMgYIABAHEB4yBggAEAcQHjIGCAAQBxAeMgYIABAHEB4yBggAEAcQHjIGCAAQBxAeMgYIABAHEB5KBAhBGABQAFieA2CSFWgAcAF4AIABkQGIAawCkgEDMi4xmAEAoAEBwAEB&sclient=gws-wiz-serp
# https://machinelearningmastery.com/gradient-descent-optimization-with-nadam-from-scratch/#:~:text=Mini%2DCourse%20now!-,Nadam%20Optimization%20Algorithm,an%20improved%20type%20of%20momentum.
# https://keras.io/api/layers/activations/
# https://towardsdatascience.com/gentle-introduction-to-selus-b19943068cd9
# https://iq.opengenus.org/scaled-exponential-linear-unit/#:~:text=in%20dying%20ReLUs.-,Advantages%20of%20SELU,functions%20without%20needing%20further%20procession.
# https://keras.io/api/layers/core_layers/dense/
# https://vitalflux.com/keras-categorical-cross-entropy-loss-function/#:~:text=categorical_crossentropy%3A%20Used%20as%20a%20loss,into%20categorical%20encoding%20using%20keras.
# https://keras.io/api/optimizers/adam/
# https://tung2389.github.io/coding-note/unitslstm
# https://datascience.stackexchange.com/questions/46388/what-is-the-difference-between-adding-more-lstm-layers-or-adding-more-units-o
# https://www.kdnuggets.com/2020/12/optimization-algorithms-neural-networks.html#:~:text=SGD%20algorithm%20is%20an%20extension,derivative%20of%20the%20loss%20function.
# https://www.google.com/search?q=lstm+best+units+explained&oq=lstm+best+units&aqs=chrome.1.69i57j33i160l2.4625j0j7&sourceid=chrome&ie=UTF-8
# https://keras.io/api/metrics/accuracy_metrics/
# https://keras.io/api/models/model_training_apis/
# https://keras.io/api/optimizers/ftrl/
# https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanSquaredError
# https://keras.io/api/losses/regression_losses/
