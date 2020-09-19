##########
#### NEURAL NET
##########

install.packages("neuralnet")
library(neuralnet)
# We will load the data from the library dataset
install.packages("dataset")
library(dataset)
# 70% of the data will be used to train the model
# 30% of the data will be used to test the model performance
n=nrow(iris)
size.train=floor(n*0.7); size.test=floor(n*0.3)
# We use this seed to be able to get the same training and test set everytime
set.seed(123)
# Definition of the observations ID assigned to the train and test data
id.train=sample(1:n,size.train,replace=FALSE)
id.test=sample(setdiff(1:n,id.train),size.test,replace=FALSE)
# We create the training and test dataset
iris_train=iris[id.train,]; iris_test=iris[id.test,]
neuralnet_model <- neuralnet((Species=="setosa") +
                               (Species=="versicolor") +
                               (Species=="virginica") ~
                               Sepal.Length+Sepal.Width+
                               Petal.Length+Petal.Width,
                             rep = 1, data = iris_train,
                             algorithm = "backprop",
                             learningrate = 0.01,
                             linear.output = FALSE, hidden = c(3, 3),
                             stepmax = 1000000, act.fct = "logistic")

plot(neuralnet_model)

neuralnet_prediction <- predict(neuralnet_model, iris_test)
table(iris_test$Species, apply(neuralnet_prediction, 1, which.max))
##########
#### KERAS
##########
##### FFNN
devtools::install_github("rstudio/keras")
library(keras)
install_keras()
library(datasets)
# Download the Iris data to workspace
data(iris)
# Change the data Iris output for string
iris$Species = sapply(as.character(iris$Species),
                      switch, "setosa" = 1,
                      "versicolor" = 2,
                      "virginica" = 3,
                      USE.NAMES = F)
# Here is another way, to split the data to
spec = c(train = .7,test = .3)
# Set a seed in order to be repreductible
set.seed(123)
# Sample through the dataframe using the sample and cut.
# The variable "g" returns a list of rows for train and
# test
g = sample(cut(seq(nrow(iris)),
               nrow(iris)*cumsum(c(0,spec)),
               labels = names(spec)
)
)
# Use the data and the row information to select rows
data_df = split(iris, g)
# Create vector that will contain X (Features variable)
# and Y the target variable
X = c()
Y = c()
X$train = as.matrix(data_df$train[,-ncol(data_df$train)])
X$test = as.matrix(data_df$test[,-ncol(data_df$test)])
Y$train = as.matrix(data_df$train[,ncol(data_df$train)])
Y$test = as.matrix(data_df$test[,ncol(data_df$test)],)

# load the library
library(keras)
# Create a sequential model
model_keras = keras_model_sequential()
# Start assembling the model with the first layer as the input.
# A softmax was use in the output because of the
model_keras %>%
  layer_dense(units = 8, activation = 'relu', input_shape = c(4)) %>%
  layer_dense(units = 4, activation = 'softmax')
summary(model_keras)
# Compile model
model_keras %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)
history <- model_keras %>% fit(
  as.matrix(X$train),to_categorical(Y$train),
  epochs=200, batch_size=5)
plot(history)
preds = predict(model_keras,as.array(X$test))
pred.label.keras <- max.col((preds)) - 1
1-mean(pred.label.keras != as.vector(data_df$test$Species))

##########
#### KERAS
##########
##### CNN
library(keras)
# We set the seed to get all the outputs reproductible.
set.seed(123)
cifar_10 = dataset_cifar10()
# RGB values are usually encoded between 0 and 255.
# A good practice is to scale them to a value from 0 and 1 b dividing the RGB values by 255.
X_train <- cifar_10$train$x/255
X_test <- cifar_10$test$x/255
# cifar_10 class labels ranging from 0 to 9 are downloaded as integer
# We will use the keras function "to_categorical" to encode them as one-hot.
Y_train <- to_categorical(cifar_10$train$y, num_classes = 10)
Y_test <- to_categorical(cifar_10$test$y, num_classes = 10)
par(mfcol=c(2,3))
par(mar=c(0, 0, 1, 0), xaxs = 'i', yaxs='i')
for (i in 1:6) {
  plot(as.raster(X_train[i,,,]))
}

# Model implementation
model <- keras_model_sequential()
model %>%
  # We start with a first 2D convolutional layer, having a kernel of size 3x3.
  # We use padding = same, meaning that our output tensor will have the same dimensions as our
  #input tensor.
  # Our input_shape is the dimension of each of our image. Here we have a 32x32x3 image (RGB).
  # For black and white images use only 1 dimension (e.g. 32x32x1)
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same",input_shape = c(32, 32, 3)) %>%
  layer_activation("relu") %>%
  layer_batch_normalization() %>%
  # We use a 2nd convolutional layer where we increase the number of kernel by 2 (32 -> 64)
  layer_conv_2d(filter = 64, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  layer_batch_normalization() %>%
  # We then use a maxpooling layer in order to reduce the dimentionnality of the
  #convolutional layer.
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  # We then flatten the maxpooling layer into a vector that will be feed into a FNN.
  # We set the first layer of our FNN to have 256 hidden neurons.
  layer_flatten() %>%
  layer_dense(256) %>%
  layer_activation("relu") %>%
  #layer_batch_normalization() %>%
  layer_dropout(0.5) %>%
  # We set the output layer of our FNN to have 10 neurons, one for each of the 10 class we
  #want to predict. We use softmax in order to scale all the output values betwen 0 and 1,
  #and that the sum of all of them is equal to 1.
  layer_dense(10) %>%
  layer_activation("softmax")

summary(model)
opt <- optimizer_rmsprop(lr = 1e-3, decay = 1e-6)
batch_size <- 64
epochs <- 5
validation <- 0.2


model %>%
  compile(loss = "categorical_crossentropy",
          optimizer = opt, metrics = "accuracy")


start_time <- Sys.time()
history_cnn_keras <- model %>% fit(
  X_train, Y_train,
  batch_size = batch_size,
  epochs = epochs,
  validation_split = validation,
  shuffle = TRUE)
end_time <- Sys.time()
time_cnn_keras = end_time - start_time

model %>% evaluate(X_test, Y_test,verbose = 0)
Y_predicted <- model %>% predict_classes(X_test)



##########
#### KERAS
##########
##### RNN
library(keras)
library(dplyr)
library(ggplot2)
library(purrr)
word_index <- dataset_imdb_word_index()
length(word_index)
# This is in order to limit the number of words
max_features = length(word_index)
# cut texts after this number of words (among top max_features most common words)
maxlen = 45

# Loads the dataset which is already split in train and test
# For some reason there is a bug. When adding the parameter Max len the function don't
# return test data if not above maxlen 180
imdb <- dataset_imdb(maxlen = maxlen, seed = 123 )
imdb_hack <- dataset_imdb(maxlen = 180 , seed = 123)
imdb$train
imdb_hack$test
# Seperate the labels and the data for train and test
c(train_data, train_labels) %<-% imdb$train
c(test_data, test_labels) %<-% imdb_hack$test
# This is the word index which can be interprate as a dictionnary betweend index to words
# (e.g.) the word "modestly" is represented by the index 20608
# by default this word to index contains 88584 words
max_features
# Convert this word to index List to a more friendly data frame
word_index_df <- data.frame(
  word = names(word_index),
  idx = unlist(word_index, use.names = FALSE),
  stringsAsFactors = FALSE
)
# The first 4 indices are reserved for 4 different specials keys
# PAD: This is to fill a sequence that is shorter than Max lenght
# START: This is to fill a sequence that is shorter than Max lenght
# PAD: This is to fill a sequence that is shorter than Max lenght
# UNK: This if for an unkown words that are not part of vocabulary
# UNUSED: This if for an unused word part of the vocabulary
word_index_df <- word_index_df %>% mutate(idx = idx + 3)
word_index_df <- word_index_df %>%
  add_row(word = "<PAD>", idx = 0)%>%
  add_row(word = "<START>", idx = 1)%>%
  add_row(word = "<UNK>", idx = 2)%>%
  add_row(word = "<UNUSED>", idx = 3)
# Sort the rows by index
word_index_df <- word_index_df %>% arrange(idx)
# A function that will decode a id an return the full review
decode_review <- function(text){
  paste(map(text, function(number) word_index_df %>%
              filter(idx == number) %>%
              select(word) %>%
              pull()),
        collapse = " ")
}
decode_review(train_data[[5]])

# This function is built in the Keras environmet and as the name indicates it
# is use for padding sequences such as our reviews. after this is done all the
# review will have the same lenght. The shorter ones will be filled with the token
# <pad>. This applies to the train and test data.
train_pad <- pad_sequences(
  train_data,
  value = word_index_df %>% filter(word == "<PAD>") %>% select(idx) %>% pull(),
  padding = "post",
  maxlen = maxlen
)
test_pad <- pad_sequences(
  test_data,
  value = word_index_df %>% filter(word == "<PAD>") %>% select(idx) %>% pull(),
  padding = "post",
  maxlen = maxlen
)
#Create a validation dataset
n = as.integer(round(nrow(train_pad)*0.7))
x_val <- train_pad[1:n, ]
partial_x_train <- train_pad[(n+1):nrow(train_pad), ]
y_val <- train_labels[1:n]
partial_y_train <- train_labels[(n+1):length(train_labels)]

model <- keras_model_sequential()
model %>%
  layer_embedding(input_dim = max_features, output_dim = 32) %>%
  layer_lstm(units = 24,
             input_shape = 32,
             batch_size = 8)%>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 1, activation = "relu")
model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = list('accuracy')
)

start_time <- Sys.time()
history_rnn_keras <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  validation_data = list(x_val, y_val),
)
end_time <- Sys.time()
time_rnn_keras = end_time - start_time
results <- model %>% evaluate(test_pad, test_labels)

plot(history_rnn_keras)


##########
#### h2o
##########
##### FFNN

install.packages("h2o")
library(h2o)
h2o.init()
library(datasets)
# Download the Iris data to workspace
data(iris)
# Change the data Iris output for string
iris$Species = sapply(as.character(iris$Species),
                      switch, "setosa" = 1,
                      "versicolor" = 2,
                      "virginica" = 3,
                      USE.NAMES = F)
# Here is another way, to split the data to
spec = c(train = .7,test = .3)
# Set a seed in order to be repreductible
set.seed(123)
# Sample through the dataframe using the sample and cut.
# The variable "g" returns a list of rows for train and
# test
g = sample(cut(seq(nrow(iris)),
               nrow(iris)*cumsum(c(0,spec)),
               labels = names(spec)
)
)
# Use the data and the row information to select rows
data_df = split(iris, g)
# Create vector that will contain X (Features variable)
# and Y the target variable
X = c()
Y = c()
X$train = as.matrix(data_df$train[,-ncol(data_df$train)])
X$test = as.matrix(data_df$test[,-ncol(data_df$test)])
Y$train = as.matrix(data_df$train[,ncol(data_df$train)])
Y$test = as.matrix(data_df$test[,ncol(data_df$test)],)

# First we need to load the library
library(h2o)
h2o.init()
# Identify predictors and response
# We this package we need to convert everything in
# h2o format by casting.
h2o_df_train = as.h2o(data_df$train)
h2o_df_test = as.h2o(data_df$test)
# Labeling the target is optional but helpful to debug
y <- "Species"
x <- setdiff(names(h2o_df_train), y)
# For binary classification, response should be a
h2o_df_train[,y] <- as.factor(h2o_df_train[,y])
h2o_df_test[,y] <- as.factor(h2o_df_test[,y])
# The deeplearning is kept simple in order to give a high level of the user with
# minimal interaction
start_time <- Sys.time()
model_h2o <- h2o.deeplearning(x = x,
                              y = y,
                              training_frame = h2o_df_train,
                              epochs = 200,
                              seed=123)
end_time <- Sys.time()
time_fnn_h2o = end_time - start_time
summary(model_h2o)
perf <- h2o.performance(model_h2o, as.h2o(data_df$test))
pred <- h2o.predict(model_h2o, as.h2o(data_df$test))
1-mean(as.vector(pred$predict) != as.vector(data_df$test$Species))



##########
#### MXNET
##########
##### FFNN
install.packages("https://s3.ca-central-1.amazonaws.com/jeremiedb/share/mxnet/CPU/mxnet.zip",
                 repos = NULL)
library(datasets)
library
data(iris)
set.seed(123)
iris$Species = as.numeric(iris$Species) - 1
spec = c(train = .7,test = .3)
g = sample(cut(seq(nrow(iris)),nrow(iris)*cumsum(c(0,spec)),labels = names(spec)))
data_df = split(iris, g)
X = c()
Y = c()
X$train = as.matrix(data_df$train[,-ncol(data_df$train)])
X$test = as.matrix(data_df$test[,-ncol(data_df$test)])
Y$train = data_df$train[,ncol(data_df$train)]
Y$test = data_df$test[,ncol(data_df$test)]

library(mxnet)
data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=8)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=3)
softmax <- mx.symbol.SoftmaxOutput(fc2, name="sm")
mx.set.seed(123)
start_time <- Sys.time()
model_mx <- mx.model.FeedForward.create(
  softmax,
  X=as.array(X$train),
  y=as.numeric(Y$train),
  ctx=mx.cpu(),
  num.round=200,
  verbose = TRUE,
  eval.metric = mx.metric.accuracy,
  array.batch.size=5,
  learning.rate=0.07)
end_time <- Sys.time()
time_fnn_mxnet = end_time - start_time
preds = predict(model_mx,as.array(X$test),array.layout = "rowmajor")
summary(model_mx)
pred.label.mxnet <- max.col(t(preds)) - 1
1-mean(pred.label.mxnet != as.vector(data_df$test$Species))
