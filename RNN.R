
save.image("C:/Users/apabl/Desktop/Git/DeepLearninginR/DeepLearninginR/RNN_save.RData")


#############################
############ KERAS
#############################
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
# PAD:    This is to fill a sequence that is shorter than Max lenght
# START:  This is to fill a sequence that is shorter than Max lenght
# PAD:    This is to fill a sequence that is shorter than Max lenght
# UNK:    This if for an unkown words that are not part of vocabulary
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


model <- keras_model_sequential()
model %>% 
  layer_embedding(input_dim = max_features, output_dim = 32) %>%
  layer_lstm(units = 24,
             input_shape = 32,
             batch_size = 8)%>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 1, activation = "relu")

model %>% summary()

model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = list('accuracy')
)


#Create a validation dataset
n = as.integer(round(nrow(train_pad)*0.7))
x_val <- train_pad[1:n, ]
partial_x_train <- train_pad[(n+1):nrow(train_pad), ]

y_val <- train_labels[1:n]
partial_y_train <- train_labels[(n+1):length(train_labels)]


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
results
