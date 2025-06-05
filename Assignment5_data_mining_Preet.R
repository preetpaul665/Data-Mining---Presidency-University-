## Assignment V
## Name - Preet Paul

## Question 2

## Loading the required package "e1071"

library(e1071)

#(a)

x1 = runif(500, min = 0, max = 1) - 0.5
x2 = runif(500, min = 0, max = 1) - 0.5
y = 1*(x1^2 - x2^2 > 0)
y1 = y
y1
y1[y1==1] = "red"
y1[y1==0] = "blue"
y1

#(b)
  
plot(x1,x2, col = y1)


#(c)

library(MASS)

lr <- glm(y ~ x1 + x2, family = "binomial")
lr
summary(lr)

# (d)

y2 = lr$fitted.values
y2[y2>0.5] = "red"
y2[y2<=0.5] = "blue"
y2

## Plotting the observations, coloured according to the fitted class labels

plot(x1,x2, col = y2)

# (e)

lr1 = glm(y ~ x1^2 + x2^2 + (x1*x2) + log((x1+0.5)/(1+x1)) + log((x2+0.5)/(1+x2)) , family = "binomial")
lr1
summary(lr1)


#(f)

y3 = lr1$fitted.values
y3[y3>0.5] = "red"
y3[y3<=0.5] = "blue"
y3

## Plotting the observations, coloured according to the fitted class labels

plot(x1,x2, col = y3)

# (g)

data1 = as.data.frame(cbind(y,x1,x2))
data1

svm1 <- svm(y~.,data=data1, kernel = "linear", type = "C-classification")
svm1
summary(svm1)

y4 = predict(svm1, newdata = cbind(x1,x2))
y4
fitted(svm1)
plot(svm1, data = data1)

# (h)

svm2 = svm(y~., data = data1, kernel = "radial", type = "C-classification")
svm2
summary(svm2)

y5 = fitted(svm2)
y5
plot(svm2, data = data1)

################################################################################

## Question 3

cat1 = runif(500)
cat2 = runif(500)
response1 = 1*((cat1^4 - cat2^4 + cat1*cat2 - cat1^2*cat2 - cat1*cat2^2)> 0)

# (a)

response2 = response1
response2[response2>0.5] = "purple"
response2[response2<=0.5] = "green"
response2

## Plotting the data

plot(cat1, cat2, col = response2)

# (b)

## Comparing the support vector classifiers for cross validation and different values of cost 

data2 <- as.data.frame(cbind(response1, cat1, cat2))
sv <- list(NULL)

svm3 <- svm(response1~., data = data2, kernel = "linear",
            type = "C-classification", cross = 10, cost = 0.1)
svm3
summary(svm3)
## Plotting

plot(svm3, data = data2)

################################################################################

## Question 4

## Loading the required package "ISLR"

library(ISLR)

auto <- ISLR::Auto
View(auto)
str(auto)
sum(is.na(auto))

# (a)

auto$classifier[auto$mpg > median(auto$mpg)] = 1
auto$classifier[auto$mpg <= median(auto$mpg)] = 0
View(auto)
auto$origin <- as.factor(auto$origin)
auto$year <- as.factor(auto$year)
auto$cylinders <- as.factor(auto$cylinders)

# (b)

## cost = 0.1

attach(auto)
fit1 <- svm(classifier ~ mpg + cylinders + displacement + horsepower + weight + acceleration + year + origin,
            data = auto, kernel = "linear", type = "C-classification", cross = 10, cost = 0.1)
fit1
summary(fit1)

fitted_val1 <- fitted(fit1)
fitted_val1

## confusion matrix

library(caret)
tab1 = table(actual = auto$classifier, predicted = fitted_val1)
tab1
confusionMatrix(tab1)

## cost = 0.5

fit2 <- svm(classifier ~ mpg + cylinders + displacement + horsepower + weight + acceleration + year + origin,
            data = auto, kernel = "linear", type = "C-classification", cross = 10, cost = 0.5)
fit2
summary(fit2)

fitted_val2 <- fitted(fit2)
fitted_val2

## confusion matrix

library(caret)
tab2 = table(actual = auto$classifier, predicted = fitted_val2)
tab2
confusionMatrix(tab2)


## cost = 1

fit3 <- svm(classifier ~ mpg + cylinders + displacement + horsepower + weight + acceleration + year + origin,
            data = auto, kernel = "linear", type = "C-classification", cross = 10, cost = 1)
fit3
summary(fit3)

fitted_val3 <- fitted(fit3)
fitted_val3

## confusion matrix

library(caret)
tab3 = table(actual = auto$classifier, predicted = fitted_val3)
tab3
confusionMatrix(tab3)

# (c)
## kernel = radial

## cost = 0.1

fit4 <- svm(classifier ~ mpg + cylinders + displacement + horsepower + weight + acceleration + year + origin,
            data = auto, kernel = "radial", type = "C-classification", cross = 10, gamma = 0.5, cost = 0.1)
fit4
summary(fit4)

fitted_val4 <- fitted(fit4)
fitted_val4

## confusion matrix

library(caret)
tab4 = table(actual = auto$classifier, predicted = fitted_val4)
tab4
confusionMatrix(tab4)


## cost = 0.5

fit5 <- svm(classifier ~ mpg + cylinders + displacement + horsepower + weight + acceleration + year + origin,
            data = auto, kernel = "radial", type = "C-classification", cross = 10, gamma = 0.5, cost = 0.5)
fit5
summary(fit5)

fitted_val5 <- fitted(fit5)
fitted_val5

## confusion matrix

library(caret)
tab5 = table(actual = auto$classifier, predicted = fitted_val5)
tab5
confusionMatrix(tab5)

## cost = 1

fit6 <- svm(classifier ~ mpg + cylinders + displacement + horsepower + weight + acceleration + year + origin,
            data = auto, kernel = "radial", type = "C-classification", cross = 10, gamma = 0.5, cost = 1)
fit6
summary(fit6)

fitted_val6 <- fitted(fit6)
fitted_val6

## confusion matrix

library(caret)
tab6 = table(actual = auto$classifier, predicted = fitted_val6)
tab6
confusionMatrix(tab6)

## kernel = ploynomial

## cost = 0.1

fit7 <- svm(classifier ~ mpg + cylinders + displacement + horsepower + weight + acceleration + year + origin,
            data = auto, kernel = "polynomial", type = "C-classification", cross = 10, gamma = 0.5, cost = 0.1, degree = 4)
fit7
summary(fit7)

fitted_val7 <- fitted(fit7)
fitted_val7

## confusion matrix

library(caret)
tab7 = table(actual = auto$classifier, predicted = fitted_val7)
tab7
confusionMatrix(tab7)


## cost = 0.5

fit8 <- svm(classifier ~ mpg + cylinders + displacement + horsepower + weight + acceleration + year + origin,
            data = auto, kernel = "polynomial", type = "C-classification", cross = 10, gamma = 0.5, cost = 0.5, degree = 4)
fit8
summary(fit8)

fitted_val8 <- fitted(fit8)
fitted_val8

## confusion matrix

library(caret)
tab8 = table(actual = auto$classifier, predicted = fitted_val8)
tab8
confusionMatrix(tab8)

## cost = 1

fit9 <- svm(classifier ~ mpg + cylinders + displacement + horsepower + weight + acceleration + year + origin,
            data = auto, kernel = "polynomial", type = "C-classification", cross = 10, gamma = 0.5, cost = 1, degree = 4)
fit9
summary(fit9)

fitted_val9 <- fitted(fit9)
fitted_val9

## confusion matrix

library(caret)
tab9 = table(actual = auto$classifier, predicted = fitted_val9)
tab9
confusionMatrix(tab6)

## Plotting

par(mfrow = c(3,3))
plot(auto$mpg, auto$horsepower, col = fitted_val1)
plot(auto$mpg, auto$horsepower, col = fitted_val2)
plot(auto$mpg, auto$horsepower, col = fitted_val3)
plot(auto$mpg, auto$horsepower, col = fitted_val4)
plot(auto$mpg, auto$horsepower, col = fitted_val5)
plot(auto$mpg, auto$horsepower, col = fitted_val6)
plot(auto$mpg, auto$horsepower, col = fitted_val7)
plot(auto$mpg, auto$horsepower, col = fitted_val8)
plot(auto$mpg, auto$horsepower, col = fitted_val9)

################################################################################

## Question 5

OJ1 <- ISLR::OJ
View(OJ1)
str(OJ1)
OJ1$StoreID <- as.factor(OJ1$StoreID)
OJ1$SpecialCH <- as.factor(OJ1$SpecialCH)
OJ1$SpecialMM <- as.factor(OJ1$SpecialMM)
OJ1$STORE <- as.factor(OJ1$STORE)
dim(OJ1)

# (a)

## Creating a training set of 800 observations

set.seed(19)
index <- sample(1:1070, size = 800)
train1 <- OJ1[index,]
test1 <- OJ1[-index,]
dim(train1)
dim(test1)

# (b)

attach(OJ1)

model1 <- svm(Purchase ~., data = train1, type = "C-classification",
              kernel = "linear", cost = 0.01)
model1
summary(model1)

fitted_val10 <- fitted(model1)
fitted_val10

## confusion matrix

tab10 = table(actual = train1$Purchase, predicted = fitted_val10)
tab10
c1 = confusionMatrix(tab10)
c1

Train_error1 = 1 - c1$overall["Accuracy"]
Train_error1

## Prediction on test set


fitted_test1 = predict(model1, newdata = test1)
fitted_test1

## confusion matrix

tab11 = table(actual = test1$Purchase, predicted = fitted_test1)
tab11
c2 = confusionMatrix(tab11)
c2

Test_error1 = 1 - c2$overall["Accuracy"]
Test_error1

# (d)

model2 = tune(svm, Purchase ~., data = train1, cost = c(seq(0.1,1,1)), kernel = "linear")
model2$best.model

## Best cost = 0.1

# (c)

model3 <- svm(Purchase ~., data = train1, type = "C-classification",
              kernel = "linear", cost = 0.1)
model3
summary(model3)

fitted_val11 <- fitted(model3)
fitted_val11

## confusion matrix

tab12 = table(actual = train1$Purchase, predicted = fitted_val11)
tab12
c3 = confusionMatrix(tab12)
c3

Train_error2 = 1 - c3$overall["Accuracy"]
Train_error2

## Prediction on test set


fitted_test2 = predict(model3, newdata = test1)
fitted_test2

## confusion matrix

tab13 = table(actual = test1$Purchase, predicted = fitted_test2)
tab13
c4 = confusionMatrix(tab13)
c4

Test_error2 = 1 - c4$overall["Accuracy"]
Test_error2

# (f)

tune(svm, Purchase ~., data = train1, cost = c(seq(0.1,1,1)), kernel = "radial")$best.model


model4 <- svm(Purchase ~., data = train1, type = "C-classification",
              kernel = "radial", cost = 0.1)
model4
summary(model4)

fitted_val12 <- fitted(model4)
fitted_val12

## confusion matrix

tab14 = table(actual = train1$Purchase, predicted = fitted_val12)
tab14
c5 = confusionMatrix(tab14)
c5

Train_error3 = 1 - c5$overall["Accuracy"]
Train_error3

## Prediction on test set


fitted_test3 = predict(model4, newdata = test1)
fitted_test3

## confusion matrix

tab15 = table(actual = test1$Purchase, predicted = fitted_test3)
tab15
c6 = confusionMatrix(tab15)
c6

Test_error3 = 1 - c6$overall["Accuracy"]
Test_error3

# (g)

tune(svm, Purchase ~., data = train1, cost = c(seq(0.1,1,1)), kernel = "polynomial", degree = 2)$best.model


model5 <- svm(Purchase ~., data = train1, type = "C-classification",
              kernel = "polynomial", cost = 0.1, degree = 2)
model5
summary(model5)

fitted_val13 <- fitted(model5)
fitted_val13

## confusion matrix

tab16 = table(actual = train1$Purchase, predicted = fitted_val13)
tab16
c7 = confusionMatrix(tab14)
c7

Train_error4 = 1 - c7$overall["Accuracy"]
Train_error4

## Prediction on test set


fitted_test4 = predict(model5, newdata = test1)
fitted_test4

## confusion matrix

tab17 = table(actual = test1$Purchase, predicted = fitted_test4)
tab17
c8 = confusionMatrix(tab17)
c8

Test_error4 = 1 - c8$overall["Accuracy"]
Test_error4





















