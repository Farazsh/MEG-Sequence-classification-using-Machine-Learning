- This Repository contains code for reproducing the results in the study "Machine Learning for detecting Auditory sequence in Magnetoencephalography data"

Following are the description of individual scripts in the project. The list in arranged in chronological order of usage

- DataPreparation : Used to preprocess the MEG data of the so-called 'task period' in which participants were presented with sound sequences.
          It can be used for the following
	- Resampling the data to a lower frequency
	- Artifacts rejection (AR)
	- Viewing the Pre and Post AR MEG data

- DataPreparationSilent: Used to perform the same tasks as in 'DataPreparation' on the 'silent data' instead,
                         i.e. when the participants sit in idle position. Additionally
			 it also contains function for creating epochs from the continuous silent data

- DataVisualization : Used to visualize topography map of the evoked and silent data. 

- ErfPlots : Used to plot the ERF (Event related fields) of the evoked data. for the silent data it plots the average

- CommonFunctions : Contains several auxillary functions to help in data analysis

- BestParameterFinder : Used for  finding the best hyperparameters through cross-validation and grid search.
		        In this project, used for finding the best parameter for Logictic Regression.

- TrainTest : Used for model training and testing. Saves the model as pickle file and the training and test results as numpy array

- AccuracyPlotter : Used to plot the accuracy curves of train, test and cv data of model

- PredictLabelProbability : Uses the trained model to predict the probability of the silent data

- HeatMapslabel : used to plot the heatmaps of the predicted labels of silent data