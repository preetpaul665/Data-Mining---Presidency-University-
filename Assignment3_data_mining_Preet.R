## Assignment 3 - Data Mining 
## Name - Preet Paul

## Question 1

## Loading the dataset from MASS package

library(MASS)

cars1 <- MASS::Cars93
View(cars1)

## Standardizing all the variables of the dataset

str(cars1)
cars1 <- cars1[, -c(1:3,9,10,11,16,26,27)]
avg = apply(cars1, 2, FUN = mean, na.rm = T)
sd1 = apply(cars1, 2, FUN = sd, na.rm = T)
## Standardized dataset
cars2 <- (cars1 - avg)/sd1
View(cars2)

#(a)
## Fit a tree based model to predict the car price with Wheelbase and Horsepower as the covariates. Make a suitable plot of the tree

library(tree)

tree1 = tree(Price ~ Wheelbase + Horsepower, data = cars2)
tree1

## Plotting the tree

plot(tree1)
text(tree1, cex = 0.75)

#(b)
##  Plot of the recursive partition of the sample space

Price.deciles = quantile(cars2$Price, probs = seq(0, 1, 0.1))

cut.price = cut(cars2$Price, Price.deciles, include.lowest = TRUE)

plot(cars2$Wheelbase, cars2$Horsepower, col = grey(10:2/10)[cut.price],
     pch = 20, xlab = "Wheelbase", ylab = "Horsepower")
partition.tree(tree1, ordvars = c("Wheelbase","Horsepower"), add = TRUE)

#(c)
##  Find the summary statistics of the tree and describe the results obtained

summary(tree1)

# Here “deviance” is just mean squared error.
# The flexibility of a tree is basically controlled by how many leaves they have,
# since that’s how many cells they partition things into. The tree fitting function
# has a number of controls settings which limit how much it will grow — each
# node has to contain a certain number of points, and adding a node has to reduce
# the error by at least a certain amount. The default for the latter, mindev, is
# 0.01; let’s turn it down and see what happens.

#(d)
## Adjust suitable parameters to fit a more complex tree

tree2 = tree(Price ~ Wheelbase + Horsepower, data = cars2, mindev = 0.001)
plot(tree2)
text(tree2, cex = 0.75)

##  Plot of the recursive partition of the sample space
plot(cars2$Wheelbase, cars2$Horsepower, col = grey(10:2/10)[cut.price],
     pch = 20, xlab = "Wheelbase", ylab = "Horsepower")
partition.tree(tree2, ordvars = c("Wheelbase","Horsepower"), add = TRUE)

##  Find the summary statistics of the tree and describe the results obtained
summary(tree2)

# We can also include all the input features in our model. Obviously that will increase the quality of fit.

#(e)
##  Improving my prediction tree using all possible covariates

tree3 = tree(Price ~., data = cars2)
plot(tree3)
text(tree3, cex = 0.75)

## Summary statistics of the tree
summary(tree3)

#(f)
## Finding the best pruned tree of size 5

tree_prune = prune.tree(tree3, best = 5)
tree_prune
## Plotting the pruned tree
plot(tree_prune)
text(tree_prune, cex = 0.75)

#(g)
## Plot the in-sample error rate for the pruned trees of different sizes

tree.prune.seq = prune.tree(tree3)
tree.prune.seq
plot(tree.prune.seq$size, tree.prune.seq$dev, type = "s", xlab = "size", ylab = "in-sample error rate",
     main = "Plot of the in-sample error rate for the pruned trees of different sizes")

#(h)
which(tree.prune.seq$dev == min(tree.prune.seq$dev))

# size = 7 is the best  best pruned structure based on this in-sample error rate 

#(i)
## Perform a 5 fold cross-validation and plot the error rates

tree.cv = cv.tree(tree3, K = 5)
tree.cv

## Plotting the error rates

plot(tree.cv)

#(j)
## Find the best pruned tree model based on this error rate

opt.tree.cv = which(tree.cv$dev == min(tree.cv$dev))
opt.tree.cv
best.leaves = min(tree.cv$size[opt.tree.cv])
best.leaves
tree_prune2 = prune.tree(tree3, best = best.leaves)
plot(tree_prune2)
text(tree_prune2, cex = 0.75)

# size = 7 is the best pruned tree based on this error rate.
# Also, it is similar to the case of in-sample error rate.

################################################################################

## Question 2

## Loading the dataset from ISLR package

library(ISLR)

OJ1 = ISLR::OJ
View(OJ1)
dim(OJ1)

#(a)
##  Create a training set containing a random sample of 800 observations, and a test set containing the remaining observations

set.seed(19)
index = sample(1:nrow(OJ1), size = 800)
training = OJ1[index, ]
testing = OJ1[-index, ]
dim(training)
dim(testing)

#(b)
##  Fit a tree to the training data, with Purchase as the response and the other variables except for Buy as predictors

library(tree)

treefit1 = tree(Purchase ~., data = OJ1)

## Producing summary statistics about the tree

summary(treefit1)

# Here, the training error rate is 0.1636

# Hence, there are 8 terminal nodes in the tree.

#(c)
##  Type in the name of the tree object in order to get a detailed text output

treefit1

#(d)
## Plotting the tree

plot(treefit1)
text(treefit1, cex = 0.75)

#(e)
##  Predict the response on the test data, and produce a confusion matrix comparing the test labels to the predicted test labels

pred1 = predict(treefit1, newdata = testing)
typeof(pred1)
dim(pred1)
prediction1 = pred1[,1]>pred1[,2]
prediction1 = as.factor(prediction1)
levels(prediction1) = c("MM", "CH")

## Confusion Matrix

tab1 = table(testing$Purchase, as.factor(prediction1))
tab1
library(caret)
confusionMatrix(data = prediction1, reference = testing$Purchase)

# Therefore, the test error rate is 1 - Accuracy = 0.1407

#(f)
## Apply cross validation to the training set in order to determine the optimal tree size

treefit.cv = cv.tree(treefit1)
treefit.cv

#(g)
## Produce a plot with tree size on the x-axis and cross-validated classification error rate on the y-axis

plot(treefit.cv)

#(h)
# The tree which has size = 8 corresponds to the lowest cross-validated classification error rate

#(i)
##  Produce a pruned tree corresponding to the optimal tree size obtained using cross-validation

opt.fit.cv = which(treefit.cv$dev == min(treefit.cv$dev))
best.leaves1 = treefit.cv$size[opt.fit.cv]
treefit.pruned = prune.tree(treefit1, best = best.leaves1)
summary(treefit.pruned)

## Plot of the pruned tree

plot(treefit.pruned)
text(treefit.pruned, cex = 0.75)

# We can clearly see that, cross-validation  does not lead to selection of a pruned tree

## Creating a pruned tree with five terminal nodes

treefit.prune1 = prune.tree(treefit1, best = 5)
summary(treefit.prune1)

## Plotting the pruned tree

plot(treefit.prune1)
text(treefit.prune1, cex = 0.75)

#(j)
# The training error rate of pruned and unpruned tree are 0.185 and 0.1636.
# Hence, the pruned tree has more training error rate.

#(k)
## Compare the test error rates between the pruned and unpruned trees

##  Predict the response on the test data, and produce a confusion matrix comparing the test labels to the predicted test labels

pred2 = predict(treefit.prune1, newdata = testing)
prediction2 = pred2[,1]>pred2[,2]
prediction2 = as.factor(prediction2)
levels(prediction2) = c("MM","CH")

tab2 = table(testing$Purchase, prediction2)
tab2
confusionMatrix(data = prediction2, reference = testing$Purchase)

## The testing error rate of pruned tree is 1 - Accuracy = 0.1741
## Also, the testing error of the unpruned tree is 0.1407.
## Hence, the pruned tree has more testing error rate.

################################################################################

## Question 3

## Loading the dataset

Boston <- MASS::Boston
View(Boston)
dim(Boston)

#(a)
## Split the data into a training set and a test set

set.seed(19)
library(caret)
index1 = createDataPartition(1:nrow(Boston), p = (2/3), list = FALSE)
training1 = Boston[index1, ]
testing1 = Boston[-index1, ]
dim(training1)
dim(testing1)

#(b)
## Fit a regression tree and prune it to the optimal size. Report the misclassification error on the test set

library(tree)
newtree1 = tree(medv ~., data = training1)

## summary of the tree
summary(newtree1)

## Plotting the tree
plot(newtree1)
text(newtree1, cex = 0.75)

## Pruning the tree

new.prune.seq = prune.tree(newtree1)
new.prune.seq
plot(new.prune.seq)  ## Plotting the deviance against different sizes

opt.new.prune = which(new.prune.seq$dev == min(new.prune.seq$dev))
best.new.leaves = new.prune.seq$size[opt.new.prune]
new.prune1 = prune.tree(newtree1, best = best.new.leaves)
summary(new.prune1)

## Plotting the pruned tree
plot(new.prune1)
text(new.prune1, cex = 0.75)

## Predicting the model on the test set

new.prune2 = prune.tree(new.prune1, best = best.new.leaves, newdata = testing1)
summary(new.prune2)

pred3 = predict(new.prune1, newdata = testing1)
pred3 = round(pred3,1)

## Plotting the tree
plot(new.prune2)
text(new.prune2, cex = 0.75)

#(c)
## Use randomForest() function from randomForest package to implement bagging

library(randomForest)

set.seed(19)
rf1 = randomForest(medv ~., data = training1)
rf1

#(d)
##  How well does this bagged model perform on the test set? Compare the test set MSE with the optimally pruned single tree

set.seed(19)
rf2 = randomForest(medv ~., data = testing1)
rf2

# Here, percentage of variation explained is 72.48, which is quite good.
# Also, the test set MSE is 21.07745 which is higher in comparison of the 
# optimally pruned single tree, which is 15.3

#(e)
## Fit a random forest of regression trees. Explore the use of mtry argument in the random forest

set.seed(19)
rf3 = randomForest(medv ~., data = training1, mtry = ncol(training1)/3)
rf3

#(f)
## How well does this random forest model perform on the test set? Compare the test set MSE with that of bagging and single tree

set.seed(19)
rf4 = randomForest(medv ~., data = testing1, mtry = ncol(testing1)/3)
rf4

# Here, percentage of variation explained is 73.79, which is quite good.
# Also, the test set MSE is 20.07224 which is higher in comparison of the single tree which is 15.3
# but, lower in comparison of the bagging tree, which is 21.07745.

#(g)
## Using the importance() function, we can view the importance of each variable. Make a plot of the importance using the inbuilt function varImpPlot(). Interpret the results

importance(rf4)

## Showing the Variable Importance plot for all the above randomforest models

par(mfrow = c(2,2))
varImpPlot(rf1, sort = TRUE, 
           main = "randomforest model on train set", col = "red")
varImpPlot(rf2, sort = TRUE, 
           main = "randomforest model on test set", col = "blue")
varImpPlot(rf3, sort = TRUE, 
           main = "randomforest model using mtry on train set", col = "red")
varImpPlot(rf4, sort = TRUE, 
           main = "randomforest model using mtry on test set", col = "blue")

# From the above four cases, we can clearly see that the variables "rm" and "lstat" have most importance.

#(h)
##  Use the boosting() function from adabag package to impelement adaptive boosting and compare the test set MSE

library(adabag)

# boost1 = boosting(medv ~., data = training1, boos = TRUE)
# boost1
# boost1$importance

#(j)
## Use the gbm() function from gbm package to implement gradient boosting and find the test set MSE

library(gbm)

set.seed(19)
gbm1 = gbm(medv ~., data = training1)
gbm1
gbm2 = gbm(medv ~., data = testing1)
MSE1 = mean((gbm2$fit - testing1$medv)^2)
MSE1

# Hence, the test set MSE is 13.29088.

## summary of the model

summary(gbm1)

## Change the value of the shrinkage parameter in gbm to investigate how it affects the final results
## Shrinkage = 0.01

gbm3 = gbm(medv ~., data = training1, shrinkage = 0.01)
gbm3
summary(gbm3)

## Shrinkage = 0.001

gbm4 = gbm(medv ~., data = training1, shrinkage = 0.001)
gbm4
summary(gbm4)

## We can clearly see that, as the shrinkage or learning rate is getting smaller,
## the variable importance of "rm" and "lstat" increases significantly and the importance of
## other variables become almost insignificant.

################################################################################

## Question 4

## Loading the dataset

spam.f1 = "C:\\Users\\PREET PAUL\\Desktop\\Presidency University M.Sc. Notes\\4th Semester\\Programming\\spambase.data"
spam = read.csv(spam.f1, header = F, sep = ",")
View(spam)
dim(spam)
str(spam)
## The last column corresponds to if the message is spam or email
sum(spam[,58]==1)  ## 1 corresponds to spam
sum(spam[,58]==0)  ## 0 corresponds to email

#(a)
##  Use the package ipred (short for Improved Predictors) for bagging decision trees

library(ipred)
library(rpart)

y = as.factor(spam[,58])
x = spam[,-58]


## Using bootstrap replications = 10

## Using decision stumps
bag11 = ipredbagg(y, x, nbagg = 10, coob = TRUE,
                    control = rpart.control(cp = 0, xval = 0, maxdepth = 1))
bag11

## Using 4-node trees
bag12 = ipredbagg(y, x, nbagg = 10, coob = TRUE,
                  control = rpart.control(cp = 0, xval = 0, maxdepth = 4))

bag12

## Using 8 node trees
bag13 = ipredbagg(y, x, nbagg = 10, coob = TRUE,
                  control = rpart.control(cp = 0, xval = 0, maxdepth = 8))

bag13

## Using largest possible trees
bag14 = ipredbagg(y, x, nbagg = 10, coob = TRUE,
                  control = rpart.control(cp = 0, xval = 0, maxdepth = 30))

bag14

## Plotting the Average OOB misclassification rate against different size trees
barplot(c("decision stumps" = bag11$err, "4-node trees" = bag12$err, 
          "8-node trees" = bag13$err, "largest possible trees" = bag14$err),
        col = c(25,26,27,28), ylab = "OOB misclassification rate")

## Using bootstrap replications = 35

## Using decision stumps
bag21 = ipredbagg(y, x, nbagg = 35, coob = TRUE,
                  control = rpart.control(cp = 0, xval = 0, maxdepth = 1))
bag21

## Using 4-node trees
bag22 = ipredbagg(y, x, nbagg = 35, coob = TRUE,
                  control = rpart.control(cp = 0, xval = 0, maxdepth = 4))

bag22

## Using 8 node trees
bag23 = ipredbagg(y, x, nbagg = 35, coob = TRUE,
                  control = rpart.control(cp = 0, xval = 0, maxdepth = 8))

bag23

## Using largest possible trees
bag24 = ipredbagg(y, x, nbagg = 35, coob = TRUE,
                  control = rpart.control(cp = 0, xval = 0, maxdepth = 30))

bag24

## Plotting the Average OOB misclassification rate against different size trees
barplot(c("decision stumps" = bag21$err, "4-node trees" = bag22$err, 
          "8-node trees" = bag23$err, "largest possible trees" = bag24$err),
        col = c(25,26,27,28), ylab = "OOB misclassification rate")

## Using bootstrap replications = 60

## Using decision stumps
bag31 = ipredbagg(y, x, nbagg = 60, coob = TRUE,
                  control = rpart.control(cp = 0, xval = 0, maxdepth = 1))
bag31

## Using 4-node trees
bag32 = ipredbagg(y, x, nbagg = 60, coob = TRUE,
                  control = rpart.control(cp = 0, xval = 0, maxdepth = 4))

bag32

## Using 8 node trees
bag33 = ipredbagg(y, x, nbagg = 60, coob = TRUE,
                  control = rpart.control(cp = 0, xval = 0, maxdepth = 8))

bag33

## Using largest possible trees
bag34 = ipredbagg(y, x, nbagg = 60, coob = TRUE,
                  control = rpart.control(cp = 0, xval = 0, maxdepth = 30))

bag34

## Plotting the Average OOB misclassification rate against different size trees
barplot(c("decision stumps" = bag31$err, "4-node trees" = bag32$err, 
          "8-node trees" = bag33$err, "largest possible trees" = bag34$err),
        col = c(25,26,27,28), ylab = "OOB misclassification rate")

## Using bootstrap replications = 85

## Using decision stumps
bag41 = ipredbagg(y, x, nbagg = 85, coob = TRUE,
                  control = rpart.control(cp = 0, xval = 0, maxdepth = 1))
bag41

## Using 4-node trees
bag42 = ipredbagg(y, x, nbagg = 85, coob = TRUE,
                  control = rpart.control(cp = 0, xval = 0, maxdepth = 4))

bag42

## Using 8 node trees
bag43 = ipredbagg(y, x, nbagg = 85, coob = TRUE,
                  control = rpart.control(cp = 0, xval = 0, maxdepth = 8))

bag43

## Using largest possible trees
bag44 = ipredbagg(y, x, nbagg = 85, coob = TRUE,
                  control = rpart.control(cp = 0, xval = 0, maxdepth = 30))

bag44

## Plotting the Average OOB misclassification rate against different size trees
barplot(c("decision stumps" = bag41$err, "4-node trees" = bag42$err, 
          "8-node trees" = bag43$err, "largest possible trees" = bag44$err),
        col = c(25,26,27,28), ylab = "OOB misclassification rate")

## Using bootstrap replications = 110

## Using decision stumps
bag51 = ipredbagg(y, x, nbagg = 110, coob = TRUE,
                  control = rpart.control(cp = 0, xval = 0, maxdepth = 1))
bag51

## Using 4-node trees
bag52 = ipredbagg(y, x, nbagg = 110, coob = TRUE,
                  control = rpart.control(cp = 0, xval = 0, maxdepth = 4))

bag52

## Using 8 node trees
bag53 = ipredbagg(y, x, nbagg = 110, coob = TRUE,
                  control = rpart.control(cp = 0, xval = 0, maxdepth = 8))

bag53

## Using largest possible trees
bag54 = ipredbagg(y, x, nbagg = 110, coob = TRUE,
                  control = rpart.control(cp = 0, xval = 0, maxdepth = 30))

bag54

## Plotting the Average OOB misclassification rate against different size trees
barplot(c("decision stumps" = bag51$err, "4-node trees" = bag52$err, 
          "8-node trees" = bag53$err, "largest possible trees" = bag54$err),
        col = c(25,26,27,28), ylab = "OOB misclassification rate")


## Plotting the average OOB misclassification rates for bagging different size trees against B

plot(c("replication 10"=bag11$err,"replication 35"=bag21$err,"replication 60"=bag31$err,
       "replication 85"=bag41$err,"replication 110"=bag51$err),type = "b",col = 25,ylab="",ylim=c(0.05,0.25))
par(new=T)
plot(c("replication 10"=bag12$err,"replication 35"=bag22$err,"replication 60"=bag32$err,
       "replication 85"=bag42$err,"replication 110"=bag52$err),type = "b",col = 26,ylab="",ylim=c(0.05,0.25))
par(new=T)
plot(c("replication 10"=bag13$err,"replication 35"=bag23$err,"replication 60"=bag33$err,
       "replication 85"=bag43$err,"replication 110"=bag53$err),type = "b",col = 27,ylab="",ylim=c(0.05,0.25))
par(new=T)
plot(c("replication 10"=bag14$err,"replication 35"=bag24$err,"replication 60"=bag34$err,
       "replication 85"=bag44$err,"replication 110"=bag54$err),type = "b",col = 28,ylab="",ylim=c(0.05,0.25))
legend("topleft",legend = c("decision stumps","4-node trees","8-node trees","largest possible trees"),
       fill = c(25,26,27,28),bty = "b")

#(b)
## Use function ada in package ada to perform discrete adaboost and report the training and test error

library(ada)
library(caret)

set.seed(19)
index2 = createDataPartition(1:length(y), p = (2/3), list = FALSE)
y_train = y[index2]
y_test = y[-index2]
x_train = as.matrix(x[index2,])
x_test = as.matrix(x[-index2,])

## For decision stump

## Iteration = 20

ada1 = ada(x = x_train, y = y_train, test.x = x_test, test.y = y_test, loss = "exponential",
           type = "discrete", iter = 20)
ada1
cat("Training error = 0.049\n")
list("Confusion matrix for the training set is",ada1$confusion)

## Incase of test set

prediction3 = predict(ada1, newdata = as.data.frame(x_test))

## Confusion Matrix
conf1 = confusionMatrix(data = prediction3, reference = y_test)
conf1
cat("Test error = ",1 - conf1$overall["Accuracy"],"\n")

## Iteration 50

ada2 = ada(x = x_train, y = y_train, test.x = x_test, test.y = y_test, loss = "exponential",
           type = "discrete", iter = 50)
ada2
cat("Training error = 0.029\n")
list("Confusion matrix for the training set is",ada1$confusion)

## Incase of test set

prediction4 = predict(ada2, newdata = as.data.frame(x_test))

## Confusion Matrix
conf2 = confusionMatrix(data = prediction4, reference = y_test)
conf2
cat("Test error = ",1 - conf2$overall["Accuracy"],"\n")

## Iteration 100

ada3 = ada(x = x_train, y = y_train, test.x = x_test, test.y = y_test, loss = "exponential",
           type = "discrete", iter = 100)
ada3
cat("Training error = 0.013\n")
list("Confusion matrix for the training set is",ada3$confusion)

## Incase of test set

prediction5 = predict(ada3, newdata = as.data.frame(x_test))

## Confusion Matrix
conf3 = confusionMatrix(data = prediction5, reference = y_test)
conf3
cat("Test error = ",1 - conf3$overall["Accuracy"],"\n")

## For 4-node tree

## Iteration = 20

ada4 = ada(x = x_train, y = y_train, test.x = x_test, test.y = y_test, loss = "exponential",
           type = "discrete", iter = 20, control = rpart.control(cp = -1, xval = 0, maxdepth = 4))
ada4
cat("Training error = 0.059\n")
list("Confusion matrix for the training set is",ada4$confusion)

## Incase of test set

prediction6 = predict(ada4, newdata = as.data.frame(x_test))

## Confusion Matrix
conf4 = confusionMatrix(data = prediction6, reference = y_test)
conf4
cat("Test error = ",1 - conf4$overall["Accuracy"],"\n")

## Iteration = 50

ada5 = ada(x = x_train, y = y_train, test.x = x_test, test.y = y_test, loss = "exponential",
           type = "discrete", iter = 50, control = rpart.control(cp = -1, xval = 0, maxdepth = 4))
ada5
cat("Training error = 0.051\n")
list("Confusion matrix for the training set is",ada5$confusion)

## Incase of test set

prediction7 = predict(ada5, newdata = as.data.frame(x_test))

## Confusion Matrix
conf5 = confusionMatrix(data = prediction7, reference = y_test)
conf5
cat("Test error = ",1 - conf5$overall["Accuracy"],"\n")

## Iteration = 100

ada6 = ada(x = x_train, y = y_train, test.x = x_test, test.y = y_test, loss = "exponential",
           type = "discrete", iter = 100, control = rpart.control(cp = -1, xval = 0, maxdepth = 4))
ada6
cat("Training error = 0.05\n")
list("Confusion matrix for the training set is",ada6$confusion)

## Incase of test set

prediction8 = predict(ada6, newdata = as.data.frame(x_test))

## Confusion Matrix
conf5 = confusionMatrix(data = prediction8, reference = y_test)
conf5
cat("Test error = ",1 - conf5$overall["Accuracy"],"\n")

# For both the cases, the training and test error are decreasing as iteration increases.
# But, both the errors decreases more in decision stumps case than in 4-node trees case.













