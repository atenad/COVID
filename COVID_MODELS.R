##############################################################################################################################################
# Title: 
# Author details: Alberto Tena del Pozo
# Contact details: atena@cimne.upc.edu
# Copyright statement:

# DataFrame info: 

##############################################################################################################################################



##############################################################################################################################################

#Install and load the required libraries
#???chwck to see if packages are installed

packages <- c("randomForest", "caTools", "factoextra", "dplyr", "ROCR", "gplots", "ggplot2", "lattice", "MASS", "caret", "pROC", "devtools", "ggbiplot", "glmnet", "lme4", "broom", "e1071", "readxl")


##############################################################################################################################################


#Import the synthetic dataset with the time frequency_features of virufy dataset freely available in https://github.com/atenad/ALS/blob/master/ALS_PHONATORY_TIME_FREQUENCY_SYNTHETIC_DATA.xlsx

path <- "C:/Users/atena/Documents/COVID_TF_SYNTHETIC_DATA.xlsx"       ### Add the path where ALS_PHONATORY_TIME_FREQUENCY_SYNTHETIC_DATA.xlsx file is saved
  
library(readxl)
covid_tf <- read_excel(path)
View(covid_tf)


my_data <- covid_tf

my_data <- as.data.frame(my_data)

my_data$Diagnostic <- as.factor(my_data$Diagnostic)

my_data$Diagnostic <- relevel(my_data$Diagnostic, c("P"))


my_data[ ,1:40] <- scale(my_data[ , c(1:40)], center= TRUE, scale=TRUE) #### centering and scaling predictors



#################Random Forest####################

set.seed(42)

control_rf <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 1,
                      number = 10
)


result_rf <- rfe( Diagnostic ~ ., data = my_data,
                   rfeControl= control_rf)

predictors <- c(predictors(result_rf), "Diagnostic")

new_data <- my_data[which(names(my_data) %in% predictors)]

new_data <- as.data.frame(new_data)


ctrl <- trainControl(method = "repeatedcv", number = 10, repeats=1, classProbs = TRUE, savePredictions = TRUE, summaryFunction=defaultSummary, sampling = "up")

set.seed(42)


rf_fit <- caret::train(Diagnostic ~ ., data = new_data,
                       method = "rf",
                       trControl = ctrl,
                       ntree = 500,
                       #preProcess = c("center","scale")
                       #metric = "ROC"
)


resample_stats_rf <- thresholder(rf_fit,
                              threshold = seq(.5, 1, by = 0.05),
                              final = TRUE, statistics = c("F1", "Accuracy", "Sensitivity", "Specificity", "Kappa")
)


resample_stats_rf


################ SVM ###########################

set.seed(42)

control <- rfeControl(functions = caretFuncs,
                      method = "repeatedcv",
                      repeats = 1,
                      number = 10
)



result_svm <- rfe( Diagnostic ~ ., data = my_data,
                   method = "svmLinear",
                   rfeControl= control_svm)


 

predictors <- c(predictors(result_svm), "Diagnostic")

new_data <- my_data[which(names(my_data) %in% predictors)]

new_data <- as.data.frame(new_data)

grid <- expand.grid(C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5))
set.seed(42)
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats=1, classProbs = TRUE, savePredictions = TRUE, summaryFunction=defaultSummary, sampling = "up")

svm_fit <- caret::train(Diagnostic ~ ., data = new_data,
                        method = "svmLinear",
                        metric="Accuracy",
                        trControl = ctrl,
                        tuneGrid = grid, 
                        maximize = TRUE
                        #preProcess = c("center","scale")
                        #metric = "F1"
)

resample_stats_svm <- thresholder(svm_fit,
                                 threshold = seq(.5, 1, by = 0.05),
                                 final = TRUE, statistics = c("F1", "Accuracy", "Sensitivity", "Specificity", "Kappa")
)


resample_stats_svm



################################### Logistic Regresion ##############

set.seed(42)

control <- rfeControl(functions = caretFuncs,
                      method = "repeatedcv",
                      repeats = 1,
                      number = 10
)



result_lr <- rfe(Diagnostic ~ ., data = my_data,
                   method = "glm",
                   rfeControl= control)

predictors <- c(predictors(result_lr), "Diagnostic")

new_data <- my_data[which(names(my_data) %in% predictors)]

new_data <- as.data.frame(new_data)

ctrl <- trainControl(method = "repeatedcv", number = 10, repeats=1, classProbs = TRUE, savePredictions = TRUE, summaryFunction=defaultSummary, sampling = "up")

set.seed(42)

lr_fit <- caret::train(Diagnostic ~ ., data = new_data,
                        method = "glm",
                        #rep=5,
                        trControl = ctrl,
                        family = binomial(link = 'logit'),
                        control=glm.control(maxit=100),
                        preProcess = c("center","scale")
                        #metric = "ROC"
)

resample_stats_lr <- thresholder(lr_fit,
                                  threshold = seq(.5, 1, by = 0.05),
                                  final = TRUE, statistics = c("F1", "Accuracy", "Sensitivity", "Specificity", "Kappa")
)


resample_stats_lr



#################################### Naive Bayes ###############################################

set.seed(42)

control <- rfeControl(functions = caretFuncs,
                      method = "repeatedcv",
                      repeats = 1,
                      number = 10
)



result_nb <- rfe(Diagnostic ~ ., data = my_data,
                   method = "nb",
                   rfeControl= control)


predictors <- c(predictors(result_nb), "Diagnostic")

new_data <- my_data[which(names(my_data) %in% predictors)]

new_data <- as.data.frame(new_data)


ctrl <- trainControl(method = "repeatedcv", number = 10, repeats=1, classProbs = TRUE, savePredictions = TRUE, summaryFunction=defaultSummary, sampling = "up")


set.seed(42)

nb_fit <- caret::train(Diagnostic ~ ., data = new_data,
                       method = "nb",
                       trControl = ctrl,
                       family = binomial,
                       preProcess = c("center","scale")
                       #metric = "ROC"
)

resample_stats_nb <- thresholder(nb_fit,
                                 threshold = seq(.5, 1, by = 0.05),
                                 final = TRUE, statistics = c("F1", "Accuracy", "Sensitivity", "Specificity", "Kappa")
)


resample_stats_nb


##################################### LDA ########################################


set.seed(42)

control <- rfeControl(functions = ldaFuncs,
                      method = "repeatedcv",
                      repeats = 1,
                      number = 10
)



result_lda <- rfe(Diagnostic ~ ., data = my_data,
                   rfeControl= control)


predictors <- c(predictors(result_lda), "Diagnostic")

new_data <- my_data[which(names(my_data) %in% predictors)]

new_data <- as.data.frame(new_data)

ctrl <- trainControl(method = "repeatedcv", number = 10, repeats=1, classProbs = TRUE, savePredictions = TRUE, summaryFunction=defaultSummary, sampling = "up")


set.seed(42)

lda_fit <- caret::train(Diagnostic ~ ., data = new_data,
                        method = "lda",
                        trControl = ctrl,
                        preProcess = c("center","scale")
                        #metric = "ROC""
)

resample_stats_lda <- thresholder(lda_fit,
                                 threshold = seq(.5, 1, by = 0.05),
                                 final = TRUE, statistics = c("F1", "Accuracy", "Sensitivity", "Specificity", "Kappa")
)


resample_stats_lda

