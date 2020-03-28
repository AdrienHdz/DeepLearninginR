#############################
############ GET IRIS DATASET
#############################
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
print(tail(X$train))
print(tail(Y$train))

#############################
############ KERAS
#############################
library(keras)

# create model
model_keras = keras_model_sequential()
model_keras %>% 
  layer_dense(units = 8, activation = 'relu', input_shape = c(4)) %>% 
  layer_dense(units = 3, activation = 'softmax')
# Compile model
model_keras %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)
start_time <- Sys.time()
history <- model_keras %>% fit(
  as.matrix(X$train),to_categorical(Y$train), 
   epochs=200, batch_size=5)
end_time <- Sys.time()

#Benchmark
time_fnn_keras = end_time - start_time
model_keras %>% evaluate(X$test, to_categorical(Y$test))
model_keras %>% predict(X$test, to_categorical(Y$test))

preds = predict(model_keras,as.array(X$test))
pred.label.keras <- max.col((preds)) - 1
1-mean(pred.label.keras != as.vector(data_df$test$Species))

#############################
############ H2o
#############################
#install.packages("h2o")
library(h2o)
h2o.init()
# Identify predictors and response


h2o_df_train = as.h2o(data_df$train)
h2o_df_test = as.h2o(data_df$test)

y <- "Species"
x <- setdiff(names(h2o_df_train), y)
# For binary classification, response should be a factor
h2o_df_train[,y] <- as.factor(h2o_df_train[,y])
h2o_df_test[,y] <- as.factor(h2o_df_test[,y])

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
pred.label.h2o <- h2o.predict(model_h2o, as.h2o(data_df$test))
acc_h2o = 1-mean(as.vector(pred.label.h2o$predict) != as.vector(data_df$test$Species))



#############################
############ MXnet
#############################
install.packages("https://s3.ca-central-1.amazonaws.com/jeremiedb/share/mxnet/CPU/mxnet.zip", repos = NULL)
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
  array.batch.size=5)
end_time <- Sys.time()
time_fnn_mxnet = end_time - start_time


preds = predict(model_mx,as.array(X$test),array.layout = "rowmajor")
summary(model_mx)
pred.label.mxnet <- max.col(t(preds)) - 1
1-mean(pred.label.mxnet != as.vector(data_df$test$Species))


#############################
############ Neuralnet
#############################

library(neuralnet)
library(dataset)
data("iris")
# On divise le dataset en 70% training et 30 % testing 
n=nrow(iris) 
size.train=floor(n*0.7) 
size.test=floor(n*0.3) 

# Définition des ID des observations assignées aux train et test data 
set.seed(123) # specification de la seed pour que les résultats puissent être reproduits exactement 
id.train=sample(1:n,size.train,replace=FALSE) 
id.test=sample(setdiff(1:n,id.train),size.test,replace=FALSE) 

# Division de l'échantillon 
iris_train=iris[id.train,] 
iris_test=iris[id.test,] 

start_time <- Sys.time()
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
end_time <- Sys.time()
time_fnn_neuralnet = end_time - start_time
plot(neuralnet_model)

neuralnet_prediction <- predict(neuralnet_model, iris_test)
neural_net_CM <- table(iris_test$Species, apply(neuralnet_prediction, 1, which.max))

############# SAVE WORKSPACE

save.image("C:/Users/adrie/OneDrive/Desktop/Git/Deep_learning_r/DeepLearninginR/FNN_save.RData")







