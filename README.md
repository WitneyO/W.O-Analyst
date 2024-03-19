#TBANLT 560 Project

Breast Cancer Classification Project

```{r}
install.packages("mlbench")
library(mlbench)
require(mlbench)
library(caret)
library(ConfusionTableR)
```

```{r}
library(tidyverse)
require(mlbench)

#load the mlbench package which has the BreastCancer data set
require(mlbench)

# if you don't have any required package, use the install.packages() command
# load the data set
data(BreastCancer)
ls(BreastCancer)
# some algorithms don't like missing values, so remove rows with missing values
BreastCancer <- na.omit(BreastCancer) 

# remove the unique identifier, which is useless and would confuse the machine learning algorithms
BreastCancer$Id <- NULL 
head(BreastCancer)
str(BreastCancer)
df2 <- data.frame(sapply(BreastCancer[1:9], function(x) as.numeric(as.character(x))))
z <- scale(df2[,1:9],center=TRUE,scale=TRUE)
```
```{r}
head(z)
library(e1071)
library(caret)
mysvm <- svm(Class ~ ., BreastCancer)
mysvm.pred <- predict(mysvm, BreastCancer)
table(mysvm.pred,BreastCancer$Class)
cmatrix.svm<-table(mysvm.pred,BreastCancer$Class)
confusionMatrix(cmatrix.svm)
```
```{r}
set.seed(150)
# partition the data
train.index <- sample(c(1:dim(BreastCancer)[1]), dim(BreastCancer)[1]*0.7)
train.df <- BreastCancer[train.index, ]
holdout.df <- BreastCancer[-train.index, ]
dim(train.df)
topredict_set<-holdout.df[2:10]                       # Removing target class 
dim(topredict_set)

```

```{r}
install.packages("klaR")
library(klaR)
```

```{r}
library(caret)
library(klaR)

mynb <- NaiveBayes(Class ~ ., BreastCancer)
str(mynb)
mynb$tables
mynb$apriori
mynb.pred <- predict(mynb,BreastCancer)
head(mynb.pred$class)
table(mynb.pred$class,BreastCancer$Class)
str(mysvm.pred)
str(mynb.pred)
```
library(nnet)
library(neuralnet)
library(caret)

str(BreastCancer)
for (i in c(1:9)){
BreastCancer[,i] <-(as.numeric(BreastCancer[,i])-min(as.numeric(BreastCancer[,i]))) /
  (max(as.numeric(BreastCancer[,i]))-min(as.numeric(BreastCancer[,i])))
}
mynnet <- neuralnet(Class ~ ., BreastCancer, hidden=c(5,4))
head(BreastCancer)
str(mynnet.pred)
str(BreastCancer)
head(BreastCancer$Class)
mynnet.pred <- predict(mynnet,BreastCancer,type="class")
head(mynnet.pred[,])
mynnetClass=ifelse(mynnet.pred[,1]>.5, "benign", "malignant")
head(mynnetClass)
bcClass=ifelse(BreastCancer$Class=="benign", 1, 0)

```

```{r}
library(MASS)

#Decision trees
library(rpart)
mytree <- rpart(Class ~ ., BreastCancer)
plot(mytree); text(mytree) 
summary(mytree)
mytree.pred <- predict(mytree,BreastCancer,type="class")
table(mytree.pred,BreastCancer$Class)

cmatrix.tree<-table(mytree.pred,BreastCancer$Class)
confusionMatrix(cmatrix.tree)


# Leave-1-Out Cross Validation (LOOCV)
ans <- numeric(length(BreastCancer[,1]))
for (i in 1:length(BreastCancer[,1])) {
  mytree <- rpart(Class ~ ., BreastCancer[-i,])
  mytree.pred <- predict(mytree,BreastCancer[i,],type="class")
  ans[i] <- mytree.pred
}
ans <- factor(ans,labels=levels(BreastCancer$Class))
table(ans,BreastCancer$Class)

```
#Decision trees
library(rpart)
mytree <- rpart(Class ~ ., BreastCancer)
plot(mytree); text(mytree) 
summary(mytree)
mytree.pred <- predict(mytree,BreastCancer,type="class")
table(mytree.pred,BreastCancer$Class)

cmatrix.tree<-table(mytree.pred,BreastCancer$Class)
confusionMatrix(cmatrix.tree)


# Leave-1-Out Cross Validation (LOOCV)
ans <- numeric(length(BreastCancer[,1]))
for (i in 1:length(BreastCancer[,1])) {
  mytree <- rpart(Class ~ ., BreastCancer[-i,])
  mytree.pred <- predict(mytree,BreastCancer[i,],type="class")
  ans[i] <- mytree.pred
}
ans <- factor(ans,labels=levels(BreastCancer$Class))
table(ans,BreastCancer$Class)

```

```{r}
#Quadratic Discriminant Analysis
library(MASS)

myqda <- qda(Class ~ ., BreastCancer)
myqda.pred <- predict(myqda, BreastCancer)
head(myqda.pred$class)
table(myqda.pred$class,BreastCancer$Class)

# Leave-1-Out Cross Validation (LOOCV)
ans2 <- numeric(length(BreastCancer[,1]))
for (i in 1:length(BreastCancer[,1])) {
  myqda <- rpart(Class ~ ., BreastCancer[-i,])
  myqda.pred <- predict(myqda,BreastCancer[i,],type="class")
  ans2[i] <- myqda.pred
}
ans2 <- factor(ans,labels=levels(BreastCancer$Class))
table(ans2,BreastCancer$Class)

```
```{r}

#Regularised Discriminant Analysis
library(klaR)
myrda <- rda(Class ~ ., BreastCancer)
myrda.pred <- predict(myrda, BreastCancer)

table(myrda.pred$class,BreastCancer$Class)

```
#Random Forests
library(randomForest)
myrf <- randomForest(Class ~ ., BreastCancer)
myrf.pred <- predict(myrf, BreastCancer)
head(myrf.pred)
table(myrf.pred, BreastCancer$Class)
cmatrix.rf<-table(myrf.pred,BreastCancer$Class)
confusionMatrix(cmatrix.rf)

```

```{r}
combine.classes<-data.frame(myrf.pred, myrda.pred$class,#myqda.pred, 
                            mytree.pred,mynnet.pred,mysvm.pred, mynb.pred$class)


```

```{r}
head(combine.classes)
head(myrf.pred)
head(myrda.pred)
combine.classes$myrf.pred<-ifelse(combine.classes$myrf.pred=="benign", 0, 1)
combine.classes[,2]<-ifelse(combine.classes[,2]=="benign", 0, 1)
combine.classes[,3]<-ifelse(combine.classes[,3]=="benign", 0, 1)
combine.classes[,4]<-ifelse(combine.classes[,4]=="benign", 0, 1)
combine.classes[,5]<-ifelse(combine.classes[,5]=="benign", 0, 1)
combine.classes[,6]<-ifelse(combine.classes[,6]=="benign", 0, 1)
str(combine.classes)
combine.cl<-combine.classes[, -c(7,8)]
majority.vote=rowSums(combine.classes[,-c(7,8)])
head(majority.vote)
combine.classes[,7]<-rowSums(combine.classes[,-c(7,8)])
combine.classes[,8]<-ifelse(combine.classes[,7]>=4, "malignant", "benign")
table(combine.classes[,8], BreastCancer$Class)

classifier1<-c(0,1,0,1,0)
classifier2<-c(1,0,0,1,0)
classifier3<-c(0,0,0,1,0)
classifier4<-c(1,1,0,0,0)
classifier5<-c(1,0,0,1,0)
combine.df<-cbind(classifier1, classifier2, classifier3, classifier4, classifier5)
rowSums(combine.df)
```












