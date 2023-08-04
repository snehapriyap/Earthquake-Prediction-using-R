rm(list=ls())
cat("\014")

setwd("")

# load in the data file
data <- read.csv("earthquake_data.csv", stringsAsFactors = FALSE)
summary(data)
str(data)

hist(data$cdi, breaks = 10)

# change cdi to factor
data$cdi <- cut(data$cdi, breaks = c(-1,3,6,10), labels = c("-1", "0", "1"))

# Visualization
hist(data$magnitude, breaks = 10)
hist(data$depth, breaks = 10)
hist(data$sig, breaks = 10)

#Checking for Outliers 
boxplot(data$magnitude, main = "Magnitude")
boxplot(data$depth, main = "Depth")
boxplot(data$sig, main = "Significance")


#split the data into testing and training data sets
set.seed(123) # for reproducible results
train <- sample(1:nrow(data), nrow(data)*(2/3))

# Use the train index set to split the dataset
#  data.train for building the model
#  data.test for testing the model
data.train <- data[train, ]   # 521 rows
data.test <- data[-train, ]   # the other 261 rows

########### Classification Tree with rpart ########### 
library(rpart)

# grow tree
fit <- rpart(cdi ~ magnitude + depth + sig + latitude + longitude,
             data = data,
             method = "class",
              cp = 0.01)
fit 

# plot the tree
library(rpart.plot)
prp(fit, type = 1, extra = 1, varlen = -10, main="Classification Tree for Earthquake") 

# vector of predicted class for each observation in data.train
pred <- predict(fit, data.train, type = "class")
# actual class of each observation in data.train
actual <- data.train$cdi

# build the "confusion matrix"
confusion.matrix <- confusionMatrix(pred, actual, positive = "1")  
confusion.matrix

# data in data.test
data.pred <- predict(fit, data.test, type = "class")
data.actual <- data.test$cdi
cm1 <- confusionMatrix(data.pred, data.actual, positive = "1")
cm1


########### Logistic Regression ########### 
logit.reg <- glm(cdi ~ magnitude + depth + sig + latitude + longitude, 
                 data = data.train, family = "binomial") 
summary(logit.reg)
plot(logit.reg)

# compute predicted probabilities for data.train
logitPredict <- predict(logit.reg, data.train, type = "response")
logitPredictClass <- cut(logitPredict, breaks = c(0.03703, 0.21527, 0.69781 ,1), labels = c("-1", "0", "1"))

# evaluate classifier on data.train
actual <- data.train$cdi
predict <- logitPredictClass
confusion.matrix <- confusionMatrix(predict, actual, positive = "1")
confusion.matrix

# compute predicted probabilities for data.test
logitPredict <- predict(logit.reg, data.test, type = "response")
logitPredictClass <- cut(logitPredict, breaks = c(0.03703, 0.21527, 0.69781 ,1), labels = c("-1", "0", "1"))

# evaluate classifier on data.test
actual <- data.test$cdi
predict <- logitPredictClass
cm2 <- confusionMatrix(predict, actual, positive = "1")
cm2

########### K-Nearest Neighbors ########### 
library(caret)

# 10-fold cross-validation
ctrl <- trainControl(method = "cv", number = 10) 
knnFit <- train(cdi ~ magnitude + depth + sig + latitude + longitude, 
                data = data.train, method = "knn", trControl = ctrl, preProcess = c("center","scale"),tuneGrid = expand.grid(k = 1:10))

knnFit

# plot the # of neighbors vs. accuracy (based on repeated cross validation)
plot(knnFit)

# Evaluate classifier performance on training data
actual <- data.train$cdi
knnPredict <- predict(knnFit, data.train)
confusion.matrix <- confusionMatrix(knnPredict, actual, positive = "1")
confusion.matrix

# Evaluate classifier performance on testing data
actual <- data.test$cdi
knnPredict <- predict(knnFit, data.test)
cm3 <- confusionMatrix(knnPredict, actual, positive = "1")
cm3

########### Naive Bayes Classifier ########### 
library(e1071)

# run naive bayes
fit.nb <- naiveBayes(cdi ~ magnitude + depth + sig + latitude + longitude, 
                     data = data.train)
fit.nb

# Evaluate Performance using Confusion Matrix
actual <- data.train$cdi
# predict class probability
nbPredict <- predict(fit.nb, data.train, type = "raw")
# predict class membership
nbPredictClass <- predict(fit.nb, data.train, type = "class")
confusion.matrix <- confusionMatrix(nbPredictClass, actual, positive = "1")
confusion.matrix

# Evaluate Performance using Confusion Matrix
actual <- data.test$cdi
# predict class probability
nbPredict <- predict(fit.nb, data.test, type = "raw")
# predict class membership
nbPredictClass <- predict(fit.nb, data.test, type = "class")
cm4 <- confusionMatrix(nbPredictClass, actual, positive = "1")
cm4

#######   Support Vector Machine   ########
# Fit an SVM model using the linear kernel
model <- svm(cdi ~ magnitude + depth + sig + latitude + longitude, 
             data = data.train, kernel = "linear")
model

# Make predictions on the train set
pred <- predict(model, data.train)
actual <- data.train$cdi
confusion.matrix <- confusionMatrix(pred, actual, positive = "1")
confusion.matrix

# Make predictions on the test set
pred <- predict(model, data.test)
actual <- data.test$cdi
cm5 <- confusionMatrix(pred, actual, positive = "1")
cm5

##### compare across different methods #### 
# compare across different methods (considering Class: -1 as the "negative" class)
result1 <- rbind(cm1$byClass[1, c("Sensitivity", "Specificity", "Balanced Accuracy")],
                 cm2$byClass[1, c("Sensitivity", "Specificity", "Balanced Accuracy")],
                 cm3$byClass[1, c("Sensitivity", "Specificity", "Balanced Accuracy")],
                 cm4$byClass[1, c("Sensitivity", "Specificity", "Balanced Accuracy")],
                 cm5$byClass[1, c("Sensitivity", "Specificity", "Balanced Accuracy")])
row.names(result1) <- c("Decision Tree", "Logistic Reg", "KNN", "Naive Bayes", "SVM")
result1

# compare across different methods (considering Class: 0 as the "neutral" class)
result2 <- rbind(cm1$byClass[2, c("Sensitivity", "Specificity", "Balanced Accuracy")],
                 cm2$byClass[2, c("Sensitivity", "Specificity", "Balanced Accuracy")],
                 cm3$byClass[2, c("Sensitivity", "Specificity", "Balanced Accuracy")],
                 cm4$byClass[2, c("Sensitivity", "Specificity", "Balanced Accuracy")],
                 cm5$byClass[2, c("Sensitivity", "Specificity", "Balanced Accuracy")])
row.names(result2) <- c("Decision Tree", "Logistic Reg", "KNN", "Naive Bayes", "SVM")
result2

# compare across different methods (considering Class: 1 as the "positive" class)
result3 <- rbind(cm1$byClass[3, c("Sensitivity", "Specificity", "Balanced Accuracy")],
                cm2$byClass[3, c("Sensitivity", "Specificity", "Balanced Accuracy")],
                cm3$byClass[3, c("Sensitivity", "Specificity", "Balanced Accuracy")],
                cm4$byClass[3, c("Sensitivity", "Specificity", "Balanced Accuracy")],
                cm5$byClass[3, c("Sensitivity", "Specificity", "Balanced Accuracy")])
row.names(result3) <- c("Decision Tree", "Logistic Reg", "KNN", "Naive Bayes", "SVM")
result3

