## Purpose: Use MEPS survey data (Public used national survey) to predict medical costs using machine learning approaches & model evaluation
## DATA SETS: h224 (2020 Full Year Consolidated), h233 (2021 Full Year Consolidated)
## PROGRAMMER: Gloria Xiang 
## Date: May 2, 2024 (last modified)

setwd("~/Desktop/thesis/data")
library(haven)
library(readxl)
library(dplyr)
library(tidyverse)
library(Matrix)
library(glmnet)
library(caret)

------------------------------------------------------------------
## data processing
h224 <- read_dta("h224.dta")
h233 <- read_dta("h233.dta")
  #CPSFAMID: FAMILY IDENTIFIER 
  #FAMIDYR:ANNUAL FAMILY IDENTIFIER 
  #DUPERSID  PERSON ID (DUID + PID)
  #DUID: PANEL # + ENCRYPTED DU IDENTIFIER 
dim(h224)
dim(h233)

  ## select variables
data1 <- h224 %>% select (DUPERSID,DUID,ADAGE42,ADSEX42,RACEV1X, HISPANX, MARRY20X, OCCCAT42, FAMINC20, FAMS1231,
                          EMPST42, REGION42, OFTSMK53, ADOFTALC42, CHDAGED,
                          BPMLDX,CHOLAGED, EMPHAGED,CHBRON31, DIABDX_M18,
                          CANCERDX, ASTHDX, ARTHDX, STRKDX, RTHLTH42,MNHLTH42,
                          COGLIM53, WLKLIM53, OBTOTV20, DVTOT20, OPTOTV20, ERTOT20,
                          IPDIS20, TOTEXP20, INSURC20,PRIAP20, PRIAU20, PRIDE20,PMEDPY42,BORNUSA )

## BORNUSA==2 means born outside of US
data_imm1 <- data1 %>% filter(BORNUSA==2)
data2 <- h233 %>% select (DUPERSID,DUID,AGE21X,SEX,RACEV1X,HISPANX,MARRY21X, 
                          OCCCAT42, FAMINC21, FAMS1231,
                          EMPST42, REGION42, OFTSMK53, SDHMALC, CHDAGED,
                          BPMLDX,CHOLAGED, EMPHAGED,CHBRON31, DIABDX_M18,
                          CANCERDX, ASTHDX, ARTHDX, STRKDX, RTHLTH42,MNHLTH42,
                          COGLIM53, WLKLIM53, OBTOTV21, DVTOT21, OPTOTV21, ERTOT21,
                          IPDIS21, TOTEXP21, INSURC21,PRIAP21, PRIAU21, PRIDE21,PMEDPY42,BORNUSA)
data_imm2 <- data2 %>% filter(BORNUSA==2)

#write.csv(total2, "~/Desktop/total_clean.csv", row.names=FALSE)

dat2020 <- read_csv("data_imm1.csv")
dat2021 <- read_csv("data_imm2.csv")

#merge two year's data with the same ID
dat2021_2 <- dat2021 %>% select (DUPERSID,DUID, TOTEXP21)
total <- merge(dat2020,dat2021_2,by=c("DUPERSID","DUID"))
------------------------------------------------------------------
## standardization
## standardization for every variable
hist(total$TOTEXP21)
#total$exp_stand<-(total$TOTEXP21-mean(total$TOTEXP21))/sd(total$TOTEXP21)

# Log Transformation for right skewed data
#total$exp_stand<-log(total$TOTEXP21)
hist(total$exp2)
summary(total$exp2)

# outcome standardization
total$exp2<-(total$TOTEXP21-mean(total$TOTEXP21))/sd(total$TOTEXP21)
total2 <- total %>% select(-TOTEXP21) 

------------------------------------------------------------------
#data cleansing: data with "Don't know" or "Refused" or "Inapplicable" would be treated as missing
# 7018 records remain after data cleansing

------------------------------------------------------------------
#descriptive analysis
summary(total$TOTEXP21)
table(total$ADSEX42)
prop.table(table(total$RACEV1X)) * 100
table(total$HISPANX)
table(total$PMEDPY42)
prop.table(table(total$PMEDPY42)) * 100

summary(total$FAMINC20)
summary(total$FAMS1231)
table(total$EMPST42)
table(total$REGION42)

------------------------------------------------------------------
## divide train/test dataset
## ML assumption: train, test data need to have same distribution 
set.seed(123)
train_ind <- sample(seq_len(nrow(total2)), size = 1390)

train <- total2[train_ind, ]
test <- total2[-train_ind, ]

------------------------------------------------------------------
## ML modeling
#Research Question 1: What are the key predictors 
#Lasso feature selection use train data
x_tr <- as.matrix(train[, -34])
y_tr <- train[, 34, drop = T]
x_te <- as.matrix(test[, -34])
y_te <- test[, 34, drop = T] 

#standardized, all predictors , only remain OBTOTV20,  IPDIS20
#not standardized, all predictors , remain HISPANX, FAMINC20, FAMS1231,INSURC20, PMEDPY42  
cv_fit_lasso <- cv.glmnet(x_tr, y_tr)
coef(cv_fit_lasso)
te_pred <- predict(cv_fit_lasso, newx = x_te)
te_error <- mean((te_pred - y_te)^2)
te_error
#plot
std_fit <- preProcess(x_tr, method = c("center", "scale")) #preProcess function
x_tr_std <- predict(std_fit, x_tr)
x_te_std <- predict(std_fit, x_te)
fit_ridge <- glmnet(x_tr_std, y_tr, alpha = 1) 
library(plotmo)
plot_glmnet(fit_ridge)


# use 5 predictors to predict health expenditure and evaluate:
#OBTOTV20, IPDIS20, FAMINC20, INSURC20, PMEDPY42
#linear model
fit_lr <- lm(exp2 ~OBTOTV20+IPDIS20+FAMINC20+INSURC20+PMEDPY42, data = train)
summary_model<-summary(rf_model)
summary_model$r.squared
summary_model$fstatistic
lmod2 <- lm(exp2 ~ 1,data=train)
anova(lmod2,fit_lr)

# predicted data
prediction <- predict(fit_lr, test)
mse <- mean((test$exp2 - prediction)^2)
sqrt(mse)

###Random forest classifier
library(randomForest)
rf_model <- randomForest(exp2~OBTOTV20+IPDIS20+FAMINC20+INSURC20+PMEDPY42, data = train)
prediction <- predict(rf_model, newdata = test)
mse <- mean((test$exp2 - prediction)^2)
sqrt(mse)

###cart
library(rpart)
cart_model <- rpart(exp2~OBTOTV20+IPDIS20+FAMINC20+INSURC20+PMEDPY42, data = train)
prediction <- predict(cart_model, newdata = test)
mse <- mean((prediction - test$exp2)^2)

###Gradient boosting
library(gbm)
gbm_model <- gbm(exp2~OBTOTV20+IPDIS20+FAMINC20+INSURC20+PMEDPY42, data = train, distribution = "gaussian", n.trees = 100, interaction.depth = 3)
prediction <- predict(gbm_model, test, n.trees = 100)
mse <- mean((prediction - test$exp2)^2)


### svr
library(e1071)
svr_model <- svm(exp2~OBTOTV20+IPDIS20+FAMINC20+INSURC20+PMEDPY42, data = train, kernel = "radial")
prediction <- predict(svr_model, test)
mse <- mean((prediction - test$exp2)^2)
sqrt(mse) 

###Neural Networks
library(neuralnet)
nn_model <- neuralnet(exp2~OBTOTV20+IPDIS20+FAMINC20+INSURC20+PMEDPY42, data = train, hidden = c(5, 3)) 
prediction <- predict(nn_model, test)
mse <- mean((prediction - test$exp2)^2)


------------------------------------------------------------------
##binary classification: the health expenditure is classified as high (>median) and low (<=median)
# 1 is high, >667.5, 0 is low,<=667.5
total$exp_class<-ifelse(total$TOTEXP21>667.5 , 1, 0)
table(total$exp_class)
total2 <- total %>% select(-TOTEXP21) 

#HISPANX, FAMINC20, BPMLDX, CHOLAGED, ARTHDX,
#OBTOTV20, DVTOT20, IPDIS20, OPTOTV20, ERTOT20, INSURC20, PMEDPY42

#logistic
model <- glm(exp_class ~ HISPANX+ FAMINC20+ BPMLDX + CHOLAGED + ARTHDX+
             OBTOTV20+ DVTOT20+ IPDIS20+ OPTOTV20+ ERTOT20+ INSURC20+ PMEDPY42,
             data = train, family = binomial)

summary(model)
pred_train_prob <- predict(model, data=test, type = 'response')
predicted_class <- ifelse(pred_train_prob >= 0.5, 1, 0) 

#pred_train_label<- ifelse( pred_train_prob>0.5, 'Yes', 'No')
#table(true=test$exp_class, ptrdicted=pred_train_label)
mean(predicted_class == test$exp_class)

#AUROC
library(pROC)
roc(test$exp_class, predicted_class)

##sensitivity: 
conf_matrix <- table(test$exp_class, predicted_class)
sensitivity <- conf_matrix[2, 2] / (conf_matrix[2, 2] + conf_matrix[2, 1])
specificity <- conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 2])

precision <- conf_matrix[2, 2] / (conf_matrix[2, 2] + conf_matrix[1, 2])
recall <- conf_matrix[2, 2] / (conf_matrix[2, 2] + conf_matrix[2, 1])
f1_score <- 2 * precision * recall / (precision + recall)


###Random forest classifier
set.seed(123)
library(randomForest)
rf_model <- randomForest(exp_class ~ HISPANX+ FAMINC20+ BPMLDX + CHOLAGED + ARTHDX+
                           OBTOTV20+ DVTOT20+ IPDIS20+ OPTOTV20+ ERTOT20+ INSURC20+ PMEDPY42,
                         data = train)
prediction <- predict(rf_model, newdata = test)
predicted_class <- ifelse(prediction >= 0.5, 1, 0) 
mean(predicted_class == test$exp_class)


varImpPlot(rf_model)
impToPlot<-importance(rf_model, scale=FALSE)
dotchart(sort(impToPlot[,1]),xlab="relative importance")


###cart
library(rpart)
cart_model <- rpart(exp_class~HISPANX+ FAMINC20+ BPMLDX + CHOLAGED + ARTHDX+
                      OBTOTV20+ DVTOT20+ IPDIS20+ OPTOTV20+ ERTOT20+ INSURC20+ PMEDPY42, data = train)
prediction <- predict(cart_model, newdata = test)
predicted_class <- ifelse(prediction >= 0.5, 1, 0) 
mean(predicted_class == test$exp_class)
argPlot <- as.data.frame(cart_model$variable.importance)


###Gradient boosting
library(gbm)
gbm_model <- gbm(exp_class~HISPANX+ FAMINC20+ BPMLDX + CHOLAGED + ARTHDX+
        OBTOTV20+ DVTOT20+ IPDIS20+ OPTOTV20+ ERTOT20+ INSURC20+ PMEDPY42, 
         data = train, distribution = "gaussian", n.trees = 100, interaction.depth = 3)
prediction <- predict(gbm_model, test, n.trees = 100)
predicted_class <- ifelse(prediction >= 0.5, 1, 0) 
mean(predicted_class == test$exp_class)
summary.gbm(gbm_model)


### svr
library(e1071)
svr_model <- svm(exp_class~HISPANX+ FAMINC20+ BPMLDX + CHOLAGED + ARTHDX+
                   OBTOTV20+ DVTOT20+ IPDIS20+ OPTOTV20+ ERTOT20+ INSURC20+ PMEDPY42,
                 data = train, kernel = "radial")
prediction <- predict(svr_model, test)
predicted_class <- ifelse(prediction >= 0.5, 1, 0) 
mean(predicted_class == test$exp_class)


###Neural Networks
library(neuralnet)
nn_model <- neuralnet(exp_class~HISPANX+ FAMINC20+ BPMLDX + CHOLAGED + ARTHDX+
                        OBTOTV20+ DVTOT20+ IPDIS20+ OPTOTV20+ ERTOT20+ INSURC20+ PMEDPY42,
                      data = train, hidden = c(5, 3)) 
prediction <- predict(nn_model, test)
predicted_class <- ifelse(prediction >= 0.5, 1, 0) 
mean(predicted_class == test$exp_class)


