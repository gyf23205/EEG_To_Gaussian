# EEG_To_Gaussian
This project includes transforamtions that gain more normality to EEG relative band power features.
The assembled fixed transformation combine multiple sub=trnasformations, each of which corresponds to on sleep stage. All the sub-transformations share basic shape.
The neural network based transformations is trained in unsuperivsed manner and is initilized with the assembled fixed transformation.
Both of their utilities have been proved by experiments.
