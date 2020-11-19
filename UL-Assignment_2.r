## Dimensionality reduction on digitally written digits and breast cancer image pixels
## Methods used: PCS, MDS and IsoMaps
library(tidyverse)
library(tidyr)
library(dplyr)
library(caret)
library(smacof)
library(RDRToolbox)

#### EDA ####
digit.dat <- read.csv("data/min_mnist.csv")
#cancer.dat <- read.csv("data/WDBC.csv")

digit.dat[["X5"]] <- as.factor(digit.dat[["X5"]])

head(digit.dat)
head(cancer.dat)

# checking for missing data (cancer.dat)
apply(as_tibble((apply(cancer.dat, 2, is.na))), 2, sum) # no missing values
anyNA(cancer.dat) # no NA values


set.seed(2020)
digit.subset <- sample(nrow(digit.dat), size = 2000) # created to be used in models as 10000 observations
                                                     # are computaionally expensive
digit.subset <- digit.dat[digit.subset,]

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
digit.pca <- princomp(digit.subset[,-1], scores = T)

# creating the scree plot
variance <- (digit.pca$sdev)^2
max.var <- round(max(variance), 1)
comps <- as.integer(c(1,2,3,4, 5, 6))
plot(comps, variance[1:6], main = "Scree Plot", xlab = "Number of Components",
     ylab = "Variance", type = "o", col = "blue", ylim = c(0, max.var))

# KNN on the digits.pca

# extracting the components and adding labels
dat.labels <- as.data.frame(digit.subset$X5)
pca.comps <- as.data.frame(digit.pca$scores[,1:6])
pca.comps <- cbind.data.frame(dat.labels, pca.comps)
names(pca.comps)[1] <- "labels"
 
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
digit.2.comps.knn <- knnOnPca(pca.comps[1:3])
digit.2.comps.knn$digit.knn.model # the knn model
digit.2.comps.knn$digit.knn.acc # model accuracy


# using 3 components
digit.3.comps.knn <- knnOnPca(pca.comps[1:4])
digit.3.comps.knn$digit.knn.model # the knn model
digit.3.comps.knn$digit.knn.acc # model accuracy

# using 4 components
digit.4.comps.knn <- knnOnPca(pca.comps[1:5])
digit.4.comps.knn$digit.knn.model # the knn model
digit.4.comps.knn$digit.knn.acc # model accuracy

# Using 5 components
digit.5.comps.knn <- knnOnPca(pca.comps[1:6])
digit.5.comps.knn$digit.knn.model # the knn model
digit.5.comps.knn$digit.knn.acc # model accuracy

# using 6 components
digit.6.comps.knn <- knnOnPca(pca.comps[1:7])
digit.6.comps.knn$digit.knn.model # the knn model
digit.6.comps.knn$digit.knn.acc # model accuracy


#### MDS on digit.dat ####
# creating proximity matrix from digit.dat
digit.dist <- dist(digit.subset)

# performing SMACOF metric MDS
digit.mds <- smacofSym(digit.dist, ndim = 6, type = "ratio")
digit.mds.comps <- as.data.frame(digit.mds$conf)
digit.mds.comps <- cbind.data.frame("labels" = digit.subset$X5, digit.mds.comps)

# run KNN and return on mds component data
# mds model and the KNN accuracies
knnOnMDS <- function(mds.dataset){
  
  # splitting the data
  mds.index <- createDataPartition(mds.dataset$labels, p = 0.8, list = F)
  mds.train <- mds.dataset[mds.index,]
  mds.test <- mds.dataset[-mds.index,]
  
  # performing KNN on the mds data
  trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)
  set.seed(2020)
  mdsData.knn <- train(labels ~ ., data = mds.train,
                        method = "knn",
                        trControl = trctrl,
                        preProcess = c("center", "scale"))
  
  # obtaining knn accuracies
  mds.preds <- predict(mdsData.knn, mds.test)
  mds.acc <- confusionMatrix(mds.preds, mds.test$labels)
  
  # create return list
  returnList <- list("trainSet" = mds.train,
                     "knnModel" = mdsData.knn,
                     "modelAcc" = mds.acc)
  return(returnList)
}

# MDS with 2 components
mds.2.comps <- knnOnMDS(digit.mds.comps[,1:3])
mds.2.comps$knnModel
mds.2.comps$modelAcc

# MDS with 3 components
mds.3.comps <- knnOnMDS(digit.mds.comps[,1:4])
mds.3.comps$knnModel
mds.3.comps$modelAcc

# MDS with 4 components
mds.4.comps <- knnOnMDS(digit.mds.comps[,1:5])
mds.4.comps$knnModel
mds.4.comps$modelAcc

# MDS with 5 components
mds.5.comps <- knnOnMDS(digit.mds.comps[,1:6])
mds.5.comps$knnModel
mds.5.comps$modelAcc

# MDS with 6 components
mds.6.comps <- knnOnMDS(digit.mds.comps[,1:7])
mds.6.comps$knnModel
mds.6.comps$modelAcc

#### IsoMap with digit data ####

# running IsoMap on the digit data and adding labels to resulting data frame
digit.iso <- Isomap(as.matrix(digit.subset), dims = 6, k = 10)
iso.comp.data <- as.data.frame(digit.iso$dim2)
iso.comp.data <- cbind.data.frame(data.labs, iso.comp.data)
names(iso.comp.data)[1] <- "labels"

# function to run knn on the resulting IsoMapped model
knnOnIso <- function(isoData){
  
  # splitting the data
  iso.index <- createDataPartition(isoData$labels, p = 0.8, list = F)
  iso.train <- isoData[iso.index,]
  iso.test <- isoData[-iso.index,]
  
  # knn on the isoMaped data with 2 dimensions
  trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)
  set.seed(2020)
  isoData.knn <- train(labels ~ ., data = iso.train,
                       method = "knn",
                       trControl = trctrl,
                       preProcess = c("center", "scale"))
  
  # obtain accuracies
  iso.preds <- predict(isoData.knn, iso.test)
  iso.acc <- confusionMatrix(iso.preds, iso.test$labels)
  
  # creating return items
  returnList <- list("isoTrain" = iso.train,
                     "knnModel" = isoData.knn,
                     "knnAcc" = iso.acc)
}

# isoMap to 2 dimensions
iso.2.comps <- knnOnIso(iso.comp.data[,1:3])
iso.2.comps$isoTrain
iso.2.comps$knnModel
iso.2.comps$knnAcc

# isoMap to 3 dims
iso.3.comps <- knnOnIso(iso.comp.data[,1:4])
iso.3.comps$isoTrain
iso.3.comps$knnModel
iso.3.comps$knnAcc

# isoMap to 4 dims
iso.4.comps <- knnOnIso(iso.comp.data[,1:5])
iso.4.comps$isoTrain
iso.4.comps$knnModel
iso.4.comps$knnAcc

# isoMap to 5 dims
iso.5.comps <- knnOnIso(iso.comp.data[,1:6])
iso.5.comps$isoTrain
iso.5.comps$knnModel
iso.5.comps$knnAcc

# isomap to 6 dims
iso.6.comps <- knnOnIso(iso.comp.data[,1:7])
iso.6.comps$isoTrain
iso.6.comps$knnModel
iso.6.comps$knnAcc
