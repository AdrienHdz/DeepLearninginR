###############################
####### CNN with KERAS ########
##### on CIFAR-10 Dataset #####
###############################

# Loading librairies
library(keras)

use_session_with_seed(42)
# Loading the dataset
cifar_10 <- dataset_cifar10()

#Scale the data between 0 and 1
X_train <- cifar_10$train$x/255
X_test <- cifar_10$test$x/255
#One hot encoding
Y_train <- to_categorical(cifar_10$train$y, num_classes = 10)
Y_test <- to_categorical(cifar_10$test$y, num_classes = 10)

# Plot the 9 first images
par(mfcol=c(2,3))
par(mar=c(0, 0, 1, 0), xaxs = 'i', yaxs='i')
for (i in 1:6) {
  plot(as.raster(X_train[i,,,]))
}
summary(model)
# Model implementation 
model <- keras_model_sequential()

model %>%
  # Start with hidden 2D convolutional layer being fed 32x32 pixel images
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same",input_shape = c(32, 32, 3)) %>%
  layer_activation("relu") %>%
  layer_batch_normalization() %>%
  
  # Second hidden layer
  layer_conv_2d(filter = 64, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  layer_batch_normalization() %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(256) %>%
  layer_activation("relu") %>%
  #layer_batch_normalization() %>%
  layer_dropout(0.5) %>%
  
  # Outputs from dense layer are projected onto 10 unit output layer
  layer_dense(10) %>%
  layer_activation("softmax")

opt <- optimizer_rmsprop(lr = 0.001, decay = 1e-6)
#opt <- optimizer_adam(r = 0.001, decay = 1e-6)
batch_size <- 64
epochs <- 10
validation <- 0.2

model %>% 
    compile(loss = "categorical_crossentropy",
            optimizer = opt, metrics = "accuracy")
  

history <- model %>% fit(
    X_train, Y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = validation,
    shuffle = TRUE)

plot(history)
 # We test the model on the test set
model %>% evaluate(X_test, Y_test,verbose = 0)

Y_predicted <- model %>% predict_classes(X_test)

True_Y <- cifar_10$test$y
True_Y <- cbind(True_Y, Class_predicted)
True_Y <- as.data.frame(True_Y)
table(True_Y$V1, True_Y$Class_predicted)
cifar_10$test$y <-  cbind(cifar_10$test$y, Y_predicted)
library(caret)
True_Y$V1 <- as.factor(True_Y$V1); True_Y$Class_predicted <- as.factor(True_Y$Class_predicted)
confusionMatrix(cifar_10$test$y, True_Y$V1)

Comparaison <- as.data.frame(Y_predicted)
Comparaison <- cbind(Comparaison, cifar_10$test$y)
Comparaison$Y_predicted <- as.factor(Comparaison$Y_predicted)
Comparaison$`cifar_10$test$y` <- as.factor(Comparaison$`cifar_10$test$y`)

library(caret)
confusionMatrix(Comparaison$Y_predicted, Comparaison$`cifar_10$test$y`)

########
######## MXNET
install.packages("https://github.com/jeremiedb/mxnet_winbin/raw/master/mxnet.zip", repos = NULL)
library(mxnet)

# Load the data
library(keras)
cifar_10 <- dataset_cifar10()
typeof(cifar_10)

#Scale the data between 0 and 1
X_train <- cifar_10$train$x/255
X_test <- cifar_10$test$x/255
#One hot encoding
Y_train <- to_categorical(cifar_10$train$y, num_classes = 10)
Y_test <- to_categorical(cifar_10$test$y, num_classes = 10)

data <- mx.symbol.Variable('data')
# Start with hidden 2D convolutional layer being fed 32x32 pixel images
conv1 <- mx.symbol.Convolution(data=data, kernel=c(3,3), num_filter=32, pad = 1)
Relu1 <- mx.symbol.Activation(data=conv1, act_type="relu")
Batch1 <- mx.symbol.BatchNorm(data=Relu1)

# Second hidden layer
# 2nd convolutional layer 5x5 kernel and 50 filters.
conv2 <- mx.symbol.Convolution(data = Batch1, kernel = c(3,3), num_filter = 64)
Relu2 <- mx.symbol.Activation(data = conv2, act_type = "relu")
Batch2 <- mx.symbol.BatchNorm(data=Relu2)

# Use max pooling
pool1 <- mx.symbol.Pooling(data = Batch2, pool_type = "max", kernel = c(2,2))
dropout1 <- mx.symbol.Dropout(data=pool1)

# Flatten max filtered output into feature vector 
flat <- mx.symbol.Flatten(data = dropout1)
fcl1 <- mx.symbol.FullyConnected(data = flat, num_hidden = 256)
Relu3 <- mx.symbol.Activation(data = fcl1, act_type = "relu")
dropout2 <- mx.symbol.Dropout(data=Relu3)
# 2nd fully connected layer
fcl2 <- mx.symbol.FullyConnected(data = dropout2, num_hidden = 10)
# Output
NN_model <- mx.symbol.SoftmaxOutput(data = fcl2)

# Set seed for reproducibility
mx.set.seed(100)

# Device used. Sadly not the GPU :-(
device <- mx.cpu()

model <- mx.model.FeedForward.create(NN_model, X = X_train, y = Y_train,
                                     ctx = device,
                                     num.round = 10,
                                     array.batch.size = 64,
                                     learning.rate = 0.001,
                                     momentum = 0.9,
                                     eval.metric = mx.metric.accuracy,
                                     epoch.end.callback = mx.callback.log.train.metric(100))
