## Dimensionality reduction on digitally written digits and breast cancer image pixels
## Methods used: PCS, MDS and IsoMaps
library(tidyverse)
library(tidyr)
library(dplyr)
library(caret)

#### EDA ####
digit.dat <- read.csv("data/min_mnist.csv")
cancer.dat <- read.csv("data/WDBC.csv")

head(digit.dat)
head(cancer.dat)

digit.dat[["X5"]] <- as.factor(digit.dat[["X5"]])

# checking for missing data (cancer.dat)
apply(as_tibble((apply(cancer.dat, 2, is.na))), 2, sum) # no missing values
anyNA(cancer.dat) # no NA values

#### performing KNN (minst data) ####

apply(as_tibble((apply(digit.dat, 2, is.na))), 2, sum) # no missing values
anyNA(digit.dat)

# split into train and test set
set.seed(2020)
train.index <- createDataPartition(digit.dat$X5, p = 0.8, list = F) # 80 20 split
dig.train <- digit.dat[train.index,]
dig.test <- digit.dat[-train.index,]

# further info for the train method. Using 10 fold cv and repeating that once 
# aggregate the validation results
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)
set.seed(2020)
digit.knn <- train(X5 ~ ., data = dig.train,
                   method = "knn",
                   trControl = trctrl,
                   preProcess = c("center", "scale"))
digit.knn

plot(digit.knn) # viewing the effects on CV of the different K values

# predictions on the test set
dig.test.preds <- predict(digit.knn, dig.test[,-1])
sum(dig.test.preds == dig.test$X5)/nrow(dig.test) # accuracy

confusionMatrix(dig.test.preds, dig.test$X5)

#### PCA on digit.dat ####
set.seed(2020)
digit.pca <- princomp(digit.dat[,-1], scores = T)
head(digit.pca$scores)

# creating the scree plot
variance <- (digit.pca$sdev)^2
max.var <- round(max(variance), 1)
comps <- as.integer(c(1,2,3,4, 5, 6))
plot(comps, variance[1:6], main = "Scree Plot", xlab = "Number of Components",
     ylab = "Variance", type = "o", col = "blue", ylim = c(0, max.var))

# KNN on the digits.pca
# function for obtaining different number of components and placing it in a data frame
getComps <- function(pca.res, num.comps, dat.labels){
  # extracting the components
  comps <- data.frame(labels = dat.labels)
  this.comps <- as.data.frame(pca.res$scores[,1:num.comps])
  comps <- cbind.data.frame(comps, this.comps)
 
  return(comps)
}

# function to run KNN on the components and then return the accuracies using confusionMatrix()
knnOnPca <- function(compData){
  # splitting data
  set.seed(2020)
  index <- createDataPartition(compData$labels, p = 0.8, list = F)
  compData.train <- compData[index,]
  compData.test <- compData[-index,]
  
  # performing KNN on the PCAed data
  trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)
  set.seed(2020)
  compData.knn <- train(labels ~ ., data = compData.train,
                         method = "knn",
                         trControl = trctrl,
                         preProcess = c("center", "scale"))
  
  compData.preds <- predict(compData.knn, compData.test)
  compData.acc <- confusionMatrix(compData.preds, compData.test$label)
  
  returnList <- list("digit.knn.model" = compData.knn,
                     "digit.knn.acc" = compData.acc)
  return(returnList)
}

# using 2 components
comp2data <- getComps(digit.pca, 2, digit.dat$X5)
digit.2.comps.knn <- knnOnPca(comp2data)
digit.2.comps.knn$digit.knn.model # the knn model
digit.2.comps.knn$digit.knn.acc # model accuracy


# using 3 components
digit.3.comp <- getComps(digit.pca, 3, digit.dat$X5)
digit.3.comps.knn <- knnOnPca(digit.3.comp)
digit.3.comps.knn$digit.knn.model # the knn model
digit.3.comps.knn$digit.knn.acc # model accuracy

# using 4 components
digit.4.comp <- getComps(digit.pca, 4, digit.dat$X5)
digit.4.comps.knn <- knnOnPca(digit.4.comp)
digit.4.comps.knn$digit.knn.model # the knn model
digit.4.comps.knn$digit.knn.acc # model accuracy

# Using 5 components
digit.5.comp <- getComps(digit.pca, 5, digit.dat$X5)
digit.5.comps.knn <- knnOnPca(digit.5.comp)
digit.5.comps.knn$digit.knn.model # the knn model
digit.5.comps.knn$digit.knn.acc # model accuracy

# using 6 components
digit.6.comp <- getComps(digit.pca, 6, digit.dat$X5)
digit.6.comps.knn <- knnOnPca(digit.6.comp)
digit.6.comps.knn$digit.knn.model # the knn model
digit.6.comps.knn$digit.knn.acc # model accuracy


#### MDS on digit.dat ####

