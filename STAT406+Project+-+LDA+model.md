

```R
#import data
dat.tr <- read.csv('~/training.csv', sep=',', header=T)
dat.te <- read.csv('~/test.csv', sep=',', header=T)

```

After running the inception_v3 architecture neural network training on the dog breed pictures, the 2048-variable bottleneck features are used as the dataset for analysis
The Linear Discriminant Analysis is done through the MASS library and the misclassification rate is 0.031 for the training set is reported with the code below.



```R
library(MASS)

#build LDA model
a.lda <- lda(Type ~ ., data=dat.tr)

#making prediction
a.lda.pr <- predict(a.lda,newdata=dat.tr[,-1])

#Comparing the prediction with the training set
prediction <- table(as.matrix(a.lda.pr),as.matrix(dat.tr[,1]))

#Calculate the misclassification rate
miss.rate <- (nrow(dat.tr)-sum(diag(prediction)))/nrow(dat.tr)
miss.rate
```

A 5 K-fold is reported to get a prediction error ~ 0.11  
While leave one out cross validation prediction error is reported to be 0.57


```R
#K-fold CV function for LDA
vlda = function(v,formula,data,cl){
  require(MASS)
  grps = cut(1:nrow(data),v,labels=FALSE)[sample(1:nrow(data))]
  pred = lapply(1:v,function(i,formula,data){
    omit = which(grps == i)
    z = lda(formula,data=data[-omit,])
    predict(z,data[omit,])
  },formula,data)
  
  wh = unlist(lapply(pred,function(pp)pp$class))
  table(wh,cl[order(grps)])
}
                     
#Predict with 5 k-fold
prediction2 <- vlda(5, Type~., dat.tr, dat.tr$Type)
miss.rate2 <- (nrow(dat.tr)-sum(diag(prediction2)))/nrow(dat.tr)
miss.rate2
                     
#Predict with LOOCV  
a.lda2 <- lda(Type ~ ., data=dat.tr, CV=TRUE)

prediction3 <- table(as.matrix(a.lda2$class),as.matrix(dat.tr[,1]))
miss.rate3 <- (nrow(dat.tr)-sum(diag(prediction3)))/nrow(dat.tr)
miss.rate3

```

Applying LDA on the test set would allow us to obtain the predicted class or probability.
A csv file with probability of each class for every image is written in the file name LDAsubmission.csv


```R
#Obtain predicted class & probabilities
pr.class <- a.lda.pr$class
pr.prob <- a.lda.pr$posterior

#Making prediction on test set
a.lda.pr3 <- predict(a.lda,newdata=dat.te[,-1])
pr.te.class <- a.lda.pr3$class
pr.te.pos <- a.lda.pr3$posterior

#Writing the prediction into a submission format given by Kaggle
testpredict <- as.matrix(pr.te.pos)
piclabel <- as.matrix(dat.te[,1])
testpredict <- cbind(piclabel,testpredict)
write.table(testpredict, file= "LDAsubmission.csv", append = FALSE, sep = ",", row.names=FALSE,col.names = TRUE)


```

LDA is suitable in this case because it takes continuous independent variable and is capable of making prediction on multiple classes of categorical dependent variable. It works by looking at the linear combination of variables which best explain the datasets. LDA model the difference between the classes of data by assuming that the conditional probability density functions are both normally distributed with mean and covariance parameters. It also makes an assumption of homoscedasticity and that the covariance have full rank. A QDA was not run on the datasets because there are more variables than number of observation in each class which deem it to be not suitable for prediction.
