################################################################################

# install.packages(
#   c(
#     "tm",
#     "SnowballC,",
#     "wordcloud",
#     "RColorBrewer",
#     "ggplot2",
#     "CORElearn",
#     "Rtsne",
#     "class",
#     "caret",
#     "e1071",
#     "parallel",
#     "doParallel",
#     "CORElearn",
#     "ipred",
#     "neuralnet",
#     "randomForest",
#     "adabag"
#   )
# )
# install.packages("openNLPmodels.en", repos="http://datacube.wu.ac.at/", type="source")

library(tm)
library(wordcloud)
library(SnowballC)
library(RColorBrewer)
library(ggplot2)
library(Rtsne)
library(class)
library(caret)
library(e1071)
library(parallel)
library(doParallel)
library(CORElearn)
library(ipred)
require(neuralnet)
library(randomForest)
library(adabag)

# set working directory
WD <- getwd()
if (!is.null(WD))
  setwd(WD)

# read data
train <-
  read.table(
    "train_data.tsv",
    sep = "\t",
    header = T,
    col.names = c("label", "text"),
    comment.char = "",
    quote = ""
  )
test <-
  read.table(
    "test_data.tsv",
    sep = "\t",
    header = T,
    col.names = c("label", "text"),
    comment.char = "",
    quote = ""
  )

train$label <- as.factor(train$label)
test$label <- as.factor(test$label)

summary(train)
summary(test)

######################## 1: PRE-PROCESSING #####################################

# function that transform string into pattern
sub_space <-
  content_transformer(function(string, pattern)
    gsub(pattern, " ", string))

# function that cleans data of all possible noises
get_clean_data <- function(data, whole_words = TRUE) {
  corpus <- Corpus(VectorSource(data))
  corpus  <-
    tm_map(corpus , content_transformer(tolower)) # transfrom to lowercase
  corpus  <-
    tm_map(corpus , removeWords, stopwords('english')) # remove stopwords
  corpus  <- tm_map(corpus , removeNumbers) # remove numbers
  corpus  <- tm_map(corpus , removePunctuation) # remove punctuation
  corpus <-
    tm_map(corpus, sub_space, "[^a-zA-Z]") # remove strange symbols
  corpus <-
    tm_map(corpus, sub_space, "http\\S*") # remove links
  corpus <-
    tm_map(corpus, sub_space, "@\\S*") # remove email
  if (whole_words) {
    corpus  <- tm_map(corpus , stemDocument)
  }
  corpus  <- tm_map(corpus , stripWhitespace)
}

corpus_train = get_clean_data(train$text)
corpus_test = get_clean_data(test$text)

# function that calculates term frequency
get_term_frequency <- function(corpus, frequency = 0) {
  term_document_matrix <- TermDocumentMatrix(corpus)
  term_frequency <- rowSums(as.matrix(term_document_matrix))
  term_frequency <-
    subset(term_frequency, term_frequency >= frequency)
  return(term_frequency)
}

# function that visualizes words order by frequency
word_cloud <- function(corpus,
                       frequency = 0,
                       min_frequency = 100,
                       color = TRUE) {
  term_document_matrix <- TermDocumentMatrix(corpus)
  term_frequency <- get_term_frequency(corpus, frequency)
  word_frequency <-
    sort(rowSums(as.matrix(term_document_matrix)), decreasing = TRUE)
  gray_levels <-
    gray((word_frequency + 10) / (max(word_frequency) + 10))
  wordcloud(
    words = names(word_frequency),
    freq = word_frequency,
    min.freq = min_frequency,
    random.order = FALSE,
    colors = if (color)
      brewer.pal(12, "Paired")
    else
      gray_levels,
  )
}

term_frequency_train <- get_term_frequency(corpus_train, 100)
qplot(seq(length(term_frequency_train)),
      sort(term_frequency_train),
      xlab = "index",
      ylab = "Frequency")
barplot(
  term_frequency_train,
  main = "Term Frequncy",
  horiz = FALSE,
  xlab = "Words",
  ylab = "Frequency",
  axisnames = FALSE
)
word_cloud(corpus_train, 250, 100, TRUE)

##################### 2: FEATURE CONSTRUCTION ##################################

# construct a document-term matrix for train data
document_term_matrix_train <-
  DocumentTermMatrix(corpus_train, control = list(weighting = weightTfIdf))
# find the most similar documents in the term-document matrix
document_term_matrix_train <-
  removeSparseTerms(document_term_matrix_train, sparse = 0.98)

# construct a document-term matrix for test data
document_term_matrix_test <-
  DocumentTermMatrix(corpus_test,
                     control = list(
                       dictionary = Terms(document_term_matrix_train),
                       weighting = weightTfIdf
                     ))

document_term_matrix_train
document_term_matrix_test

mat_train = as.matrix(document_term_matrix_train)
mat_test = as.matrix(document_term_matrix_test)[, names(as.data.frame(mat_train))]

# visualize train data with PCA (Principle Components Analysis)
pca_train <- prcomp(mat_train)
pca_component_one <- as.numeric(pca_train$x[, 1])
pca_component_two <- as.numeric(pca_train$x[, 2])
# visualize the possible separation in 2D orthogonal space
qplot(pca_component_one, pca_component_two,
      color = train$label) + labs(color = 'label')

# visualize train data with t-SNE (t-Distributed Stochastic Neighbor Embedding)
tsne_train <-
  Rtsne(
    mat_train,
    perplexity = 1500,
    theta = 0.5,
    dims = 2,
    check_duplicates = FALSE
  )
tsne_train <- tsne_train$Y
tsne_component_one <- as.numeric(tsne_train[, 1])
tsne_component_two <- as.numeric(tsne_train[, 2])
# visualize the possible separation in 2D orthogonal space
qplot(tsne_component_one, tsne_component_two,
      color = train$label) + labs(color = 'label')

########################### 3: MODELING ########################################

data_train <-
  cbind(as.data.frame(mat_train), train$label)
names(data_train)[ncol(data_train)] <- "label_train"
data_test <- cbind(as.data.frame(mat_test), test$label)
names(data_test)[ncol(data_test)] <- "label_test"

# function that calculates classification accuracy
CA <- function(observed, predicted) {
  table <- table(observed, predicted)
  return(sum(diag(table)) / sum(table))
}

# k-Nearest Neighbour Classification
r_train <- which(names(data_train) == "label_train")
r_test <- which(names(data_test) == "label_test")
predicted_knn <-
  knn(data_train[, -r_train], data_test[, -r_test], train$label)
observed_knn <- test$label
# classification accuracy
ca_knn <- CA(observed_knn, predicted_knn)
ca_knn

# Decision Tree
model_dt <- CoreModel(label_train ~ ., data_train, model = "tree")
predicted_dt <- predict(model_dt, data_test, type = "class")
observed_dt <- test$label
# classification accuracy
ca_dt <- CA(observed_dt, predicted_dt)
ca_dt

# Naive Bayes
model_nb <- CoreModel(label_train ~ ., data_train, model = "bayes")
predicted_nb <- predict(model_nb, data_test, type = "class")
observed_nb <- test$label
# classification accuracy
ca_nb <- CA(observed_nb, predicted_nb)
ca_nb

# Support Vector Machine
set.seed(42)
# calculate the number of cores
nuber_of_cores <- detectCores() - 1
# create the cluster for caret to use
cluster <- makePSOCKcluster(nuber_of_cores)
registerDoParallel(cluster)
# svm with a linear kernel
model_svm <-
  train(as.factor(label_train) ~ .,
        data = data_train,
        method = "svmLinear")
predicted_svm <-
  predict(model_svm, data_test, type = "raw")
observed_svm <- test$label
# classification accuracy
ca_svm <- CA(observed_svm, predicted_svm)
ca_svm
stopCluster(cluster)
registerDoSEQ()

# Artificial Neural Networks
set.seed(42)
model_nn = neuralnet(
  label_train ~ .,
  data_train,
  hidden = c(5, 10),
  threshold = 0.5,
  learningrate = 0.0001,
  rep = 10
)
predicted_nn <- compute(model_nn, data_test)$net.result[, 2]
predicted_nn <- ifelse(predicted_nn > 0.5, 1, 0)
observed_nn <- test$label
# classification accuracy
ca_nn <- CA(observed_nn, predicted_nn)
ca_nn
plot(model_nn)

# Bagging
model_ba <- bagging(label_train ~ ., data_train, nbagg = 12)
predicted_ba <- predict(model_ba, data_test, type = "class")$class
observed_ba <- test$label
# classification accuracy
ca_ba <- CA(observed_ba, predicted_ba)
ca_ba

# Random forest (variation of bagging)
model_rf <- randomForest(label_train ~ ., data_train)
predicted_rf <- predict(model_rf, data_test, type = "class")
observed_rf <- test$label
# classification accuracy
ca_rf <- CA(observed_rf, predicted_rf)
ca_rf

# Boosting
model_bo <- boosting(label_train ~ ., data_train)
predicted_bo <- predict(model_bo, data_test)$class
observed_bo <- test$label
# classification accuracy
ca_bo <- CA(observed_bo, predicted_bo)
ca_bo

# conclusion
performances <-
  c(ca_knn, ca_dt, ca_nb, ca_svm, ca_nn, ca_ba, ca_rf, ca_bo)
algorithm_names <-
  c("KNN", "DT", "NB", "SVM", "NN", "BA", "RF", "BO")
conclusion <- data.frame(performances, algorithm_names)
conclusion <-
  conclusion[rev(order(conclusion$performances)),]
rownames(conclusion) <- NULL
conclusion

########################## 4: EVALUATION #######################################

# function that calculates classification accuracy
CA <- function(observed, predicted) {
  table <- table(observed, predicted)
  return(sum(diag(table)) / sum(table))
}

# function that calculates f1 score of a model
CF1 <- function(observed, predicted) {
  confusion_matrix <-
    confusionMatrix(predicted, observed, mode = "everything", positive = "1")
  return(confusion_matrix$byClass['F1'])
}

# calculate majority
majority <- as.vector(table(train$label))
majority_class <- if (majority[1] < majority[2]) 1 else 0
observed_maj <- test$label
predicted_maj <- test$label
predicted_maj[predicted_maj != majority_class] <- majority_class
ca_maj <- CA(observed_maj, predicted_maj)
ca_maj
f1_maj <- CF1(observed_maj, predicted_maj)
f1_maj

# calculate performance
ca_knn <- CA(observed_knn, predicted_knn)
ca_knn
f1_knn <- CF1(observed_knn, predicted_knn)
f1_knn
ca_dt <- CA(observed_dt, predicted_dt)
ca_dt
f1_dt <- CF1(observed_dt, predicted_dt)
f1_dt
ca_nb <- CA(observed_nb, predicted_nb)
ca_nb
f1_nb <- CF1(observed_nb, predicted_nb)
f1_nb
ca_svm <- CA(observed_svm, predicted_svm)
ca_svm
f1_svm <- CF1(observed_svm, predicted_svm)
f1_svm
ca_nn <- CA(observed_nn, predicted_nn)
ca_nn
f1_nn <- CF1(observed_nn, as.factor(predicted_nn))
f1_nn
ca_ba <- CA(observed_ba, predicted_ba)
ca_ba
f1_ba <- CF1(observed_ba, as.factor(predicted_ba))
f1_ba
ca_rf <- CA(observed_rf, predicted_rf)
ca_rf
f1_rf <- CF1(observed_rf, predicted_rf)
f1_rf
ca_bo <- CA(observed_bo, predicted_bo)
ca_bo
f1_bo <- CF1(observed_bo, as.factor(predicted_bo))
f1_bo

# conclusion
accuracy_ca <-
  c(ca_maj, ca_knn, ca_dt, ca_nb, ca_svm, ca_nn, ca_ba, ca_rf, ca_bo)
accuracy_f1 <-
  c(f1_maj, f1_knn, f1_dt, f1_nb, f1_svm, f1_nn, f1_ba, f1_rf, f1_bo)
accuracy_ca_f1 <-
  c(
    ca_maj,
    f1_maj,
    ca_knn,
    f1_knn,
    ca_dt,
    f1_dt,
    ca_nb,
    f1_nb,
    ca_svm,
    f1_svm,
    ca_nn,
    f1_nn,
    ca_ba,
    f1_ba,
    ca_rf,
    f1_rf,
    ca_bo,
    f1_bo
  )
algorithm_names <-
  c("Majority", "KNN", "DT", "NB", "SVM", "NN", "BA", "RF", "BO")
ensemble_model <- as.factor(c(0, 0, 0, 0, 0, 0, 1, 1, 1))

conclusion <-
  data.frame(accuracy_ca, accuracy_f1, algorithm_names, ensemble_model)
conclusion <-
  conclusion[rev(order(conclusion$accuracy_f1)),]
rownames(conclusion) <- NULL
conclusion

positions <- as.vector(conclusion$algorithm_names)
ggplot(data = conclusion,
       aes(x = algorithm_names, y = accuracy_ca, color = ensemble_model)) +
  geom_point(size = 4, shape = 4) +
  scale_x_discrete(limits = positions) +
  ylim(0.42, 1) +
  xlab("Ensemble type") +
  ylab("Accuracy") +
  geom_hline(yintercept = max(accuracy_ca), color = "darkgreen") +
  geom_hline(yintercept = min(accuracy_ca), color = "black") +
  title("Performance comparison") +
  geom_text(label = positions,
            nudge_x = 0.2,
            nudge_y = -0.01) +
  theme_bw() +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  )

# compare ca with f1
graph_data <-
  data.frame(
    type = rep(c("CA", "F1")),
    performances = accuracy_ca_f1,
    algorithm_names = rep(algorithm_names, each = 2)
  )
ggplot(data = graph_data,
       aes(x = algorithm_names, y = performances, fill = type)) +
  geom_bar(stat = "identity", position = position_dodge())

# differences between ca and f1
graph_data <-
  data.frame(differences = abs(accuracy_ca - accuracy_f1), algorithm_names)
ggplot(data = graph_data, aes(x = algorithm_names, y = differences)) +
  geom_bar(stat = "identity")

################################################################################
