

#############################
############ KERAS
#############################
library(keras)


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



