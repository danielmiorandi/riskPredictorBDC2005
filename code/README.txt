This folder contains all the source code necessary for executing the RiskPredictor.
- featuresExtractor.py : reads raw data from the data/ folder and creates a corresponding features matrix in the features/ folder
- labeller.py : starting from the raw data in data/ creates labels for training the model. Results are stored in features/
- trainClassifier.py : trains the classifiers (200 20-features random forests) based on labels and features. The resulting model is saved in models/
- validator.py: performs a leave-one-out cross-validation of the model(s) developed
- predictor.py: this file takes relevant data streams and outputs the predicted risk level (H,M,L). As for the time being historical data from data/ is used, it is sufficient to indicate a date/time/position and the relevant data will be fetchted by the script.