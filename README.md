# Tutorial on Deep Learning in R: Project Overview
For the course of Avanced Statistical Learning in HEC, I created with my colleague a guided tutorial for deep learning in R. We tried as much as possible to test the latest packagez and best practices in this field.

* Created a tutorial with 30 pages called **report.pdf** to help other students building quick deep learning models in R
* Did an extansive research of 11 R tools, packages and API to build the most used deep learning models (FNN, CNN, RNN), and explained their pros and cons
* Explained the methodology to follow when building deep learning models in R
* Created 6 examples of model creation from scratch
* Gave a comparaison of the computation time between the different packages 

## Code and Resources Used
**R Version:** 3.6.2
**Packages:** tensorflow, keras, rtorch, h2o, mxnet, rnn, brnn, deepnet, neuralnettools, nnet, neuralnet
**datasets:** Iris, CIFAR-10, IMDB, 
**Course description:** https://www.hec.ca/en/courses/detail/?cours=MATH80619A 

## Workflow for building a model we recommand
* Collection, pre-processing and cleaning of the data. Partition of the data: training/test/validation; cross-validation)
* Choose the right model according to the task to perform. Example: CNN for image classification
* Train the selected models with the training data by checking the validation accuracy evolving through the training process
* Modify model's hyperparameters and experiment changes in the model architecture in order to improve accuracy
* Evaluate and compare your models on the test data 

 
    
