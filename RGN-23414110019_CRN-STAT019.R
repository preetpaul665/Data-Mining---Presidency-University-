## Name - Preet Paul
## Registration Number - 23414110019

## Question 2

## Loading the required packages 

library(keras)
library(reticulate)

## Loading the dataset in R

cifar <- dataset_cifar10()
x_train <- cifar$train$x
y_train <- cifar$train$y
x_test <- cifar$test$x
y_test <- cifar$test$y

## Checking the dimensions

dim(x_train)
dim(x_test)
dim(y_train)
dim(y_test)


## Normalizing the image pixel values in both training and test set

x_train <- x_train/255
x_test <- x_test/255

## Converting the labels in both the set to categorical values

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

model <- keras_model_sequential() %>%

# Convolutional layer 1
	layer_conv_2d(filters = 32, kernel_size = c(3,3), input_shape = c(32,32,3),
				activation = "relu", padding = "same") %>%
	layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu") %>%
	layer_max_pooling_2d(pool_size = c(2,2)) %>%
	layer_dropout(rate = 0.25) %>%

# Convolutional layer 2
	layer_conv_2d(filters = 64, kernel_size = c(3,3),
			activation = "relu", padding = "same") %>%
	layer_conv_2d(filters = 64, kernel_size = c(3,3),
			activation = "relu") %>%
	layer_max_pooling_2d(pool_size = c(2,2)) %>%
	layer_dropout(rate = 0.25) %>%

# Flatten & Dense layers
	layer_flatten() %>%
	layer_dense(units = 512, activation = "relu") %>%
	layer_dropout(rate = 0.5) %>%
	layer_dense(units = 10, activation = "softmax")

## Compiling the model

model %>% compile(
	optimizer = optimizer_adam(),
	loss = "categorical_crossentropy",
	metrics = "accuracy")

## Summarizing the model

summary(model)

## Fitting the model using training data

history = model %>% fit(
		x_train, y_train,
		epoch = 20,
		batch_size = 128,
		validation_split = 0.2)

plot(history)

## Reporting the test set loss & accuracy

model %>% evaluate(x_test, y_test)

############################################################################

## Question 1

exam <- load("C:\\Users\\user\\Desktop\\STAT019\\Exam.Rdata")
exam
dataset <- as.data.frame(dataset)
View(dataset)
str(dataset)

dim(dataset)
dim(pairwise.distance)

# (a)

dataset$Crime
pairwise.distance <- as.data.frame(pairwise.distance)
View(pairwise.distance)

## Performing Unsupervised Learning methods

## Performing Hierarchical Clustering using hclust()

dim(pairwise.distance)
d1 = dist(pairwise.distance)
View(d1)
model1 = hclust(d1, method = "complete")
model1

## Plotting the Hierarchical Clustering

plot(model1, main = "cluster diagram of Hierarchical clustering")
rect.hclust(model1, k = 3, border = "red")
cut1 = cutree(model1, k = 3)  ## Cut Interpretation
table(cut1)

# (b)
dim(dataset)
## Executing the given R code

colnames(dataset)
colnames(dataset) = c("Crime",2:93)
index = which(is.na(dataset[,1]))
test = dataset[index,]
train = dataset[-index,]
View(train)
train$Crime = as.factor(train$Crime)

## Performing Supervised Learning methods

## Performing SVM using svm()

## Loading the required package "e1071"

library(e1071)

attach(train)
fit1 = svm(Crime~., data = train, type = "C-classification",
		kernel = "radial")
fit1
summary(fit1)

## Predicting classes on test set

prediction1 = predict(fit1, newdata = test[,-1])
prediction1

## Performing random forest using randomForest()


library(randomForest)

fit2 = randomForest(Crime~., data = train)
fit2

## Plotting the variable importances

importance(fit2)
varImpPlot(fit2)

## Predicting classes on test set

prediction1 = predict(fit2, newdata = test[,-1])
prediction1
























