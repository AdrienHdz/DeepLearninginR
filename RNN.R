
save.image("C:/Users/apabl/Desktop/Git/DeepLearninginR/DeepLearninginR/RNN_save.RData")
#############################
############ KERAS
#############################
library(keras)
library(dplyr)
library(ggplot2)
library(purrr)
max_features = 20000
# cut texts after this number of words (among top max_features most common words)
maxlen = 80
imdb <- dataset_imdb(num_words = max_features)
c(train_data, train_labels) %<-% imdb$train
c(test_data, test_labels) %<-% imdb$test
word_index <- dataset_imdb_word_index()


word_index_df <- data.frame(
  word = names(word_index),
  idx = unlist(word_index, use.names = FALSE),
  stringsAsFactors = FALSE
)

# The first indices are reserved  
word_index_df <- word_index_df %>% mutate(idx = idx + 3)
word_index_df <- word_index_df %>%
  add_row(word = "<PAD>", idx = 0)%>%
  add_row(word = "<START>", idx = 1)%>%
  add_row(word = "<UNK>", idx = 2)%>%
  add_row(word = "<UNUSED>", idx = 3)

word_index_df <- word_index_df %>% arrange(idx)

decode_review <- function(text){
  paste(map(text, function(number) word_index_df %>%
              filter(idx == number) %>%
              select(word) %>% 
              pull()),
        collapse = " ")
}

decode_review(train_data[[1]])

train_data <- pad_sequences(
  train_data,
  value = word_index_df %>% filter(word == "<PAD>") %>% select(idx) %>% pull(),
  padding = "post",
  maxlen = maxlen
)

test_data <- pad_sequences(
  test_data,
  value = word_index_df %>% filter(word == "<PAD>") %>% select(idx) %>% pull(),
  padding = "post",
  maxlen = maxlen
)

model <- keras_model_sequential()
model %>% 
  layer_embedding(input_dim = max_features, output_dim = 256) %>%
  layer_lstm(units = 128,
             input_shape = 256,
             batch_size = 256)%>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% summary()

model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = list('accuracy')
)


#Create a validation dataset

x_val <- train_data[1:10000, ]
partial_x_train <- train_data[10001:nrow(train_data), ]

y_val <- train_labels[1:10000]
partial_y_train <- train_labels[10001:length(train_labels)]

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 2,
  validation_data = list(x_val, y_val),
)

results <- model %>% evaluate(test_data, test_labels)

#############################
############ MXnet
#############################
download.data <- function(data_dir) {
  dir.create(data_dir, showWarnings = FALSE)
  if (!file.exists(paste0(data_dir,'input.txt'))) {
    download.file(url='https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tinyshakespeare/input.txt',
                  destfile=paste0(data_dir,'input.txt'), method='wget')
  }
}

make_data <- function(path, seq.len = 32, dic=NULL) {
  
  text_vec <- read_file(file = path)
  text_vec <- stri_enc_toascii(str = text_vec)
  text_vec <- str_replace_all(string = text_vec, pattern = "[^[:print:]]", replacement = "")
  text_vec <- strsplit(text_vec, '') %>% unlist
  
  if (is.null(dic)) {
    char_keep <- sort(unique(text_vec))
  } else char_keep <- names(dic)[!dic == 0]
  
  # Remove terms not part of dictionary
  text_vec <- text_vec[text_vec %in% char_keep]
  
  # Build dictionary
  dic <- 1:length(char_keep)
  names(dic) <- char_keep
  
  # reverse dictionary
  rev_dic <- names(dic)
  names(rev_dic) <- dic
  
  # Adjust by -1 to have a 1-lag for labels
  num.seq <- (length(text_vec) - 1) %/% seq.len
  
  features <- dic[text_vec[1:(seq.len * num.seq)]] 
  labels <- dic[text_vec[1:(seq.len*num.seq) + 1]]
  
  features_array <- array(features, dim = c(seq.len, num.seq))
  labels_array <- array(labels, dim = c(seq.len, num.seq))
  return (list(features_array = features_array, labels_array = labels_array, dic = dic, rev_dic = rev_dic))
}

seq.len <- 100
download.data("./")
data_prep <- make_data(path = "input.txt", seq.len = seq.len, dic=NULL)



