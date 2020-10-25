#### Dimension reduction on cancer data ####

library(tidyverse)
library(tidyr)
library(dplyr)
library(caret)
library(smacof)
library(RDRToolbox)

cancer.dat <- read.csv("data/WDBC.csv")
cancer.dat$diagnosis <- as.factor(cancer.dat$diagnosis)
#cancer.dat <- as_tibble(cancer.dat)

# checking for missing data (cancer.dat)
apply(as_tibble((apply(cancer.dat, 2, is.na))), 2, sum) # no missing values
anyNA(cancer.dat) # no NA values

#### Performing KNN on raw data ####

# split the data
set.seed(2020)
train.index <- createDataPartition(cancer.dat$diagnosis, p = 0.8, list = F) # 80 20 split
cancer.train <- cancer.dat[train.index,]
cancer.test <- cancer.dat[-train.index,]

# further info for the train method. Using 10 fold cv and repeating that once 
# aggregate the validation results
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)
set.seed(2020)
cancer.knn <- train(diagnosis ~ ., data = cancer.train[,-1],
                   method = "knn",
                   trControl = trctrl,
                   preProcess = c("center", "scale"))
cancer.knn

plot(cancer.knn) # viewing the effects on CV of the different K values

# predictions on the test set
cancer.test.preds <- predict(cancer.knn, cancer.test[,-c(1,2)])
sum(cancer.test.preds == cancer.test$diagnosis)/nrow(cancer.test) # accuracy

confusionMatrix(as.factor(cancer.test.preds), as.factor(cancer.test$diagnosis))

#### performing pca on the data and running knn on result ####
cancer.pca <- princomp(cancer.dat[,-c(1,2)], scores = T)

# extracting components
cancer.labs <- as.data.frame(cancer.dat$diagnosis)
pca.comps <- as.data.frame(cancer.pca$scores[,1:6])
pca.comps <- cbind.data.frame(cancer.labs, pca.comps)
names(pca.comps)[1] <- "diagnosis"

# function to run KNN on the components and then return the accuracies using confusionMatrix()
knnOnPca <- function(compData){
  # splitting data
  set.seed(2020)
  index <- createDataPartition(compData$diagnosis, p = 0.8, list = F)
  compData.train <- compData[index,]
  compData.test <- compData[-index,]
  
  # performing KNN on the PCAed data
  trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)
  set.seed(2020)
  compData.knn <- train(diagnosis ~ ., data = compData.train,
                        method = "knn",
                        trControl = trctrl,
                        preProcess = c("center", "scale"))
  
  compData.preds <- predict(compData.knn, compData.test)
  compData.acc <- confusionMatrix(as.factor(compData.preds), as.factor(compData.test$diagnosis))
  
  returnList <- list("cancer.knn.model" = compData.knn,
                     "cancer.knn.acc" = compData.acc)
  return(returnList)
}

# running knn ... 
# 2 components
# using 2 components
cancer.2.comps.knn <- knnOnPca(pca.comps[1:3])
cancer.2.comps.knn$cancer.knn.model # the knn model
cancer.2.comps.knn$cancer.knn.acc # model accuracy

# 3 components
cancer.3.comps.knn <- knnOnPca(pca.comps[1:4])
cancer.3.comps.knn$cancer.knn.model # the knn model
cancer.3.comps.knn$cancer.knn.acc # model accuracy

# 4 components
cancer.4.comps.knn <- knnOnPca(pca.comps[1:5])
cancer.4.comps.knn$cancer.knn.model # the knn model
cancer.4.comps.knn$cancer.knn.acc # model accuracy

# 5 components
cancer.5.comps.knn <- knnOnPca(pca.comps[1:6])
cancer.5.comps.knn$cancer.knn.model # the knn model
cancer.5.comps.knn$cancer.knn.acc # model accuracy

# 6 components
cancer.6.comps.knn <- knnOnPca(pca.comps[1:7])
cancer.6.comps.knn$cancer.knn.model # the knn model
cancer.6.comps.knn$cancer.knn.acc # model accuracy

#### MDS on cancer data ####
# generating proximity data
cancer.dist <- dist(cancer.dat)

# performing SMACOF MDS
cancer.mds <- smacofSym(cancer.dist, ndim = 6, type = "ratio")
cancer.mds.comps <- as.data.frame(cancer.mds$conf)
cancer.mds.comps <- cbind.data.frame("diagnosis" = cancer.dat$diagnosis, cancer.mds.comps)
cancer.mds.comps

# run KNN and return on mds component data
# mds model and the KNN accuracies
knnOnMDS <- function(mds.dataset){
  
  # splitting the data
  mds.index <- createDataPartition(mds.dataset$diagnosis, p = 0.8, list = F)
  mds.train <- mds.dataset[mds.index,]
  mds.test <- mds.dataset[-mds.index,]
  
  # performing KNN on the mds data
  trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)
  set.seed(2020)
  mdsData.knn <- train(diagnosis ~ ., data = mds.train,
                       method = "knn",
                       trControl = trctrl,
                       preProcess = c("center", "scale"))
  
  # obtaining knn accuracies
  mds.preds <- predict(mdsData.knn, mds.test)
  mds.acc <- confusionMatrix(as.factor(mds.preds), as.factor(mds.test$diagnosis))
  
  # create return list
  returnList <- list("trainSet" = mds.train,
                     "knnModel" = mdsData.knn,
                     "modelAcc" = mds.acc)
  return(returnList)
}

# MDS with 2 components
mds.2.comps <- knnOnMDS(cancer.mds.comps[,1:3])
mds.2.comps$knnModel
mds.2.comps$modelAcc

# MDS with 3 components
mds.3.comps <- knnOnMDS(cancer.mds.comps[,1:4])
mds.3.comps$knnModel
mds.3.comps$modelAcc

# MDS with 4 components
mds.4.comps <- knnOnMDS(cancer.mds.comps[,1:5])
mds.4.comps$knnModel
mds.4.comps$modelAcc

# MDS with 5 components
mds.5.comps <- knnOnMDS(cancer.mds.comps[,1:6])
mds.5.comps$knnModel
mds.5.comps$modelAcc

# MDS with 6 components
mds.6.comps <- knnOnMDS(cancer.mds.comps[,1:7])
mds.6.comps$knnModel
mds.6.comps$modelAcc


#### IsoMap with cancer data ####

# running IsoMap on the cancer data and adding labels to resulting data frame
cancer.iso <- Isomap(as.matrix(cancer.dat[,-c(1,2)]), dims = 6, k = 10)
iso.comp.data <- as.data.frame(cancer.iso$dim6)
iso.comp.data <- cbind.data.frame(cancer.labs, iso.comp.data)
names(iso.comp.data)[1] <- "diagnosis"

# function to run knn on the resulting IsoMapped model
knnOnIso <- function(isoData){
  
  # splitting the data
  iso.index <- createDataPartition(isoData$diagnosis, p = 0.8, list = F)
  iso.train <- isoData[iso.index,]
  iso.test <- isoData[-iso.index,]
  
  # knn on the isoMaped data with 2 dimensions
  trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)
  set.seed(2020)
  isoData.knn <- train(diagnosis ~ ., data = iso.train,
                       method = "knn",
                       trControl = trctrl,
                       preProcess = c("center", "scale"))
  
  # obtain accuracies
  iso.preds <- predict(isoData.knn, iso.test)
  iso.acc <- confusionMatrix(as.factor(iso.preds), as.factor(iso.test$diagnosis))
  
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

# isoMap to 3 dimensions
iso.3.comps <- knnOnIso(iso.comp.data[,1:4])
iso.3.comps$isoTrain
iso.3.comps$knnModel
iso.3.comps$knnAcc

# isoMap to 4 dimensions
iso.4.comps <- knnOnIso(iso.comp.data[,1:5])
iso.4.comps$isoTrain
iso.4.comps$knnModel
iso.4.comps$knnAcc

# isoMap to 5 dimensions
iso.5.comps <- knnOnIso(iso.comp.data[,1:6])
iso.5.comps$isoTrain
iso.5.comps$knnModel
iso.5.comps$knnAcc

# isoMap to 6 dimensions
iso.6.comps <- knnOnIso(iso.comp.data[,1:7])
iso.6.comps$isoTrain
iso.6.comps$knnModel
iso.6.comps$knnAcc




