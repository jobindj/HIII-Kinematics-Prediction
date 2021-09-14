Convolutional_Autoencoder.ipynb  -  First step Import the FE simulation displacement data 4D in the form of  (samples ,nodes,3 DOF,time states) which is convert to 3D  (samples,3*DOF,time states)  . Second step Train the displacement data using Convolutional Autoencoder ,from this we get Latent space which is low dimension vector.

Regression_FFNN.ipynb   - In this file , Dataset preparation for regression ,Load the saved Autoencoder model , find a regression function between FE parameter(input) and Latent space from Autoencoder(output) using Feed forward neural network

Regression_GPR.ipynb   - In this file, Dataset preparation for regression,Load the saved Autoencoder model , find a regression function between FE parameter(input) and Latent space from Autoencoder(output) using Gaussian process regression.

Regression_RandomForest.ipynb - In this file, Dataset preparation for regression,Load the saved Autoencoder model , find a regression function between FE parameter(input) and Latent space from Autoencoder(output) using Random forest regressor.
