## Instalamos paquetes ##
rm(list=ls())
require(pacman)
p_load(tidyverse, caret, rio, 
       modelsummary, 
       gamlr,        
       class, skimr,dplyr, glmnet)
library(readxl)
setwd("C:/Users/Santiago Becerra/Desktop/Santiago/Andes/Materias y Trabajos/Octavo Semestre/Big Data/Trabajo final")
ocupados <- read.csv2("area_ocupados.csv")


## Limpiamos NA´s de la variable de interés ##

sum(is.na(ocupados$P6920))
ocupados$P6920 [is.na(ocupados$P6920)] <- 0
ocupados <- subset(ocupados, P6920!=0)
table(ocupados$P6920)
ocupados <- subset(ocupados, P6920!=3)
table(ocupados$P6920)

sum(is.na(ocupados$P6440))

sum(is.na(ocupados$P6780))
table(ocupados$P6780)
ocupados$P6780 [is.na(ocupados$P6780)] <- 0
ocupados <- subset(ocupados, P6780!=0)

sum(is.na(ocupados$P6800))

sum(is.na(ocupados$P7040))

sum(is.na(ocupados$P7180))
table(ocupados$P7180)
ocupados$P7180 [is.na(ocupados$P7180)] <- 0
ocupados <- subset(ocupados, P7180!=0)

sum(is.na(ocupados$INGLABO))
table(ocupados$INGLABO)
ocupados$INGLABO [is.na(ocupados$INGLABO)] <- 0
ocupados <- subset(ocupados, INGLABO!=0)


ocupados$P6920 <- ifelse(ocupados$P6920 == 2, 1, 0)
## hacemos la partición de test y training ##
set.seed(129)
split1 <- createDataPartition(ocupados$P6920, p = .7)[[1]]
length(split1)

testing <- ocupados[-split1,]
training <- ocupados[split1,]

## Seguimos con la clasificación ##
#Las variables que se escogen para la predicción son: Tiene algún tipo de contrato (P6440), ¿El trabajo es ocasional, estacional, permanente u otro? (P6780), Horas trabajadas a la semana (P6800), ¿La semana pasada tuvo segundo trabajo u ocupación? (P7040), ¿Está afiliado a una asociación gremial o sindical? (P7180), Ingresos laborales (INGLABO)
#Variable a predecir: ¿Cotiza de fondo de pensiones? (P6920)

### Clasificación ###
######## Lasso-logit ########
X <- model.matrix(P6920 ~ P6440 + P6780 + P6800 + P7040 + P7180 + INGLABO, data = training)
Y <- training$P6920

#lamda por cv 
set.seed(3840)
modelo.cv<- cv.glmnet(X, Y, alpha=1, family = "binomial")  
modelo.cv
plot(modelo.cv)

minimo.l <- modelo.cv$lambda.min
minimo.l

#lasso con lamda mínimo
lasso.modelo <- glmnet(X,Y, family = "binomial", alpha=1, lambda = minimo.l, preProcess= c("center","scale"))
#coeficientes
lasso.modelo$beta 

#predicciones
x.test <- model.matrix(P6920 ~ P6440 + P6780 + P6800 + P7040 + P7180 + INGLABO, data = testing)
lasso.predecido <- predict(lasso.modelo, newx =x.test, type="response")
lasso.predecido
predecir.lasso <- ifelse(lasso.predecido > 0.5, 1, 0)
predecir.lasso

#Performance del modelo
library(ROCR)
lasso_testing<-data.frame(testing,predecir.lasso)
lasso_testing<-rename(lasso_testing, prediccion_lasso =s0)
with(lasso_testing,prop.table(table(P6920,prediccion_lasso))) #7.05% son falsos negativos

pred_lasso <- prediction(as.numeric(predecir.lasso), as.numeric(lasso_testing$P6920))
roc_lasso <- performance(pred_lasso,"tpr","fpr")
plot(roc_lasso, main = "ROC curve", colorize = T)
abline(a = 0, b = 1)
auc_lasso <- performance(pred_lasso, measure = "auc")
auc_lasso@y.values[[1]] #AUC=0.7214193


confusionMatrix(as.factor(lasso_testing$prediccion_lasso),as.factor(lasso_testing$P6920)) #Accuracy: 0.9051, sensitivity: 0.455, specificity: 0.987


######## Ridge-reg ########
ridge.modelo <- glmnet(x = X, y = Y, alpha = 0, nlambda = 100, standardize = TRUE)
set.seed(500)
ridge.error <- cv.glmnet(x = X, y = Y,alpha = 0, nfolds = 10, type.measure = "mse", standardize  = TRUE)
plot(ridge.error)
ridge_minimo.l<-ridge.error$lambda.min
ridge_minimo.l

#ridge con lambda óptimo
ridge.modelo <- glmnet(x = X, y = Y, alpha = 0,lambda  = ridge_minimo.l, standardize = TRUE)

#Coeficientes 
ridge.modelo$beta

#Predicciones
x.test <- model.matrix(P6920 ~ P6440 + P6780 + P6800 + P7040 + P7180 + INGLABO, data = testing)
ridge.predecido <- predict(ridge.modelo, newx =x.test,type="response")
ridge.predecido
predecir.ridge <- ifelse(ridge.predecido > 0.5, 1, 0)
predecir.ridge

#Performance del modelo
ridge_testing <- data.frame(testing,predecir.ridge)

ridge_testing <- rename(ridge_testing, prediccion_ridge =s0)
with(ridge_testing,prop.table(table(P6920,prediccion_ridge))) #2.5% son falsos negativos

pred_ridge <- prediction(as.numeric(predecir.ridge), as.numeric(ridge_testing$P6920))
roc_ridge <- performance(pred_ridge,"tpr","fpr")
plot(roc_ridge, main = "ROC curve", colorize = T)
abline(a = 0, b = 1)
auc_ridge <- performance(pred_ridge, measure = "auc")
auc_ridge@y.values[[1]] #AUC=0.58

confusionMatrix(as.factor(ridge_testing$prediccion_ridge),as.factor(ridge_testing$P6920)) #Accuracy: 0.8688, sensitivity: 0.1617, specificity: 0.998



######## Logit-reg con lambda = 0 ########
set.seed(9103)
logit.modelo <- glmnet(X,Y, family = "binomial", alpha=1, lambda = 0, preProcess=c("center","scale"))

#Coeficientes
logit.modelo$beta 

#Predicciones
x.test <- model.matrix(P6920 ~ P6440 + P6780 + P6800 + P7040 + P7180 + INGLABO, data = testing)
logit.predecido <- predict(logit.modelo, newx =x.test,type="response")
logit.predecido
predecir.logit <- ifelse(logit.predecido > 0.5, 1, 0)
predecir.logit

#Performance del modelo
logit_testing<-data.frame(testing,predecir.logit)

logit_testing<-rename(logit_testing, prediccion_logit =s0)
with(logit_testing,prop.table(table(P6920, prediccion_logit))) #7.05% son falsos negativos

pred_logit<-prediction(as.numeric(predecir.logit), as.numeric(logit_testing$P6920))
roc_logit <- performance(pred_logit,"tpr","fpr")
plot(roc_logit, main = "ROC curve", colorize = T)
abline(a = 0, b = 1)
auc_logit <- performance(pred_logit, measure = "auc")
auc_logit@y.values[[1]] ##AUC=0.7210
confusionMatrix(as.factor(logit_testing$prediccion_logit),as.factor(logit_testing$P6920)) #Accuracy: 0.904, sensitivity: 0.455, specificity: 0.986


##Remuestreo upsampling para lasso##

training$P6920 <- factor(training$P6920)
set.seed(2242)
uptraining <- upSample(x = training, y = training$P6920, yname = "Cotizante")

dim(training)
dim(uptraining)
table(uptraining$P6920)

up_X<- model.matrix(P6920 ~ P6440 + P6780 + P6800 + P7040 + P7180 + INGLABO, data = uptraining)
up_Y<- uptraining$P6920

#lambda por cv
set.seed(1233)
modelo.cv.up<- cv.glmnet(up_X, up_Y, alpha=1, family = "binomial")  
modelo.cv.up
plot(modelo.cv.up)

minimo.l.up <- modelo.cv.up$lambda.min
minimo.l.up

#lasso con lambda minimo
lasso.modelo.up <- glmnet(up_X,up_Y, family = "binomial", alpha=1, lambda = minimo.l.up, preProcess= c("center","scale"))
#coeficientes
lasso.modelo.up$beta 

#Predicciones
x.test.up <- model.matrix(P6920 ~ P6440 + P6780 + P6800 + P7040 + P7180 + INGLABO, data = testing)
lasso.up.predecido <- predict(lasso.modelo.up, newx =x.test.up, type="response")
lasso.up.predecido
predecir.lasso.up <- ifelse(lasso.up.predecido > 0.5, 1, 0)
predecir.lasso.up

#Performance del modelo
lasso.up_testing <- data.frame(testing,predecir.lasso.up)

lasso.up_testing <- rename(lasso.up_testing, prediccion_lasso.up =s0)
with(lasso.up_testing,prop.table(table(P6920, prediccion_lasso.up))) #Un 11.1% son falsos negativos

pred_lasso.up<-prediction(as.numeric(predecir.lasso.up), as.numeric(lasso.up_testing$P6920))
roc_lasso.up <- performance(pred_lasso.up,"tpr","fpr")
plot(roc_lasso.up, main = "ROC curve", colorize = T)
abline(a = 0, b = 1)
auc_lasso.up <- performance(pred_lasso.up, measure = "auc")
auc_lasso.up@y.values[[1]] #AUC=0.796

confusionMatrix(as.factor(lasso.up_testing$prediccion_lasso.up),as.factor(lasso.up_testing$P6920)) #Accuracy: 0.849, sensitivity: 0.7191, specificity: 0.8729


##Remuestreo upsampling para ridge##
set.seed(3410)
modelo.cv.ridgeup<- cv.glmnet(up_X, up_Y, alpha=0, family = "binomial")  
modelo.cv.ridgeup
plot(modelo.cv.ridgeup)

minimo.l.ridgeup <- modelo.cv.ridgeup$lambda.min
minimo.l.ridgeup 

#ridge con lambda minimo
ridge.modelo.up <- glmnet(up_X,up_Y, family = "binomial", alpha=0, lambda = minimo.l.ridgeup, preProcess= c("center","scale"))
#coeficientes 
ridge.modelo.up$beta 

#predicciones
x.test.ridgeup <- model.matrix(P6920 ~ P6440 + P6780 + P6800 + P7040 + P7180 + INGLABO, data = testing)
ridge.up.predecido <- predict(ridge.modelo.up, newx =x.test.ridgeup, type="response")
ridge.up.predecido
predecir.ridge.up <- ifelse(ridge.up.predecido > 0.5, 1, 0)
predecir.ridge.up

#Performance del modelo
ridge.up_testing<-data.frame(testing,predecir.ridge.up)

ridge.up_testing<-rename(ridge.up_testing, prediccion_ridge.up =s0)
with(ridge.up_testing,prop.table(table(P6920,prediccion_ridge.up))) ##Un 10.01% son falsos negativos

pred_ridge.up<-prediction(as.numeric(predecir.ridge.up), as.numeric(ridge.up_testing$P6920))
roc_ridge.up <- performance(pred_ridge.up,"tpr","fpr")
plot(roc_ridge.up, main = "ROC curve", colorize = T)
abline(a = 0, b = 1)
auc_ridge.up <- performance(pred_ridge.up, measure = "auc")
auc_ridge.up@y.values[[1]] #AUC=0.7707

confusionMatrix(as.factor(ridge.up_testing$prediccion_ridge.up),as.factor(ridge.up_testing$P6920)) #Accuracy: 0.856, sensitivity: 0.646, specificity: 0.894


##Remuestreo downsampling para lasso##
set.seed(0192)
downtraining <- downSample(x = training, y = training$P6920, yname = "Cotizante")

dim(training)
dim(downtraining)
table(downtraining$P6920)

down_X<- model.matrix(P6920 ~ P6440 + P6780 + P6800 + P7040 + P7180 + INGLABO, data = downtraining)
down_Y<- downtraining$P6920

#lambda por cv
set.seed(2917)
modelo.cv.down<- cv.glmnet(down_X, down_Y, alpha=1, family = "binomial")  
modelo.cv.down
plot(modelo.cv.down)

minimo.l.down <- modelo.cv.down$lambda.min
minimo.l.down 

#lasso con lambda mínimo
lasso.model.down <- glmnet(down_X,down_Y, family = "binomial", alpha=1, lambda = minimo.l.down, preProcess= c("center","scale"))
#coeficientes
lasso.model.down$beta 

#Predicciones
x.test.down <- model.matrix(P6920 ~ P6440 + P6780 + P6800 + P7040 + P7180 + INGLABO, data = testing)
lasso.down.predecido <- predict(lasso.model.down, newx =x.test.down, type="response")
lasso.down.predecido
predecir.lasso.down <- ifelse(lasso.down.predecido > 0.5, 1, 0)
predecir.lasso.down

#Performance del modelo
lasso.down_testing<-data.frame(testing,predecir.lasso.down)

lasso.down_testing<-rename(lasso.down_testing, prediccion_lasso.down =s0)
with(lasso.down_testing,prop.table(table(P6920,prediccion_lasso.down))) # 10.5 son falsos negativos

pred_lasso.down<-prediction(as.numeric(predecir.lasso.down), as.numeric(lasso.down_testing$P6920))
roc_down <- performance(pred_lasso.down,"tpr","fpr")
plot(roc_down, main = "ROC curve", colorize = T)
abline(a = 0, b = 1)
auc_down <- performance(pred_lasso.down, measure = "auc")
auc_down@y.values[[1]] #AUC=0.783

confusionMatrix(as.factor(lasso.down_testing$prediccion_lasso.down),as.factor(lasso.down_testing$P6920)) #Accuracy: 0.853, sensitivity: 0.68, specificity: 0.885

##Remuestreo downsampling para ridge##
#Lambda por cv
set.seed(572)
modelo.cv.ridgedown<- cv.glmnet(down_X, down_Y, alpha=0, family = "binomial")  
modelo.cv.ridgedown
plot(modelo.cv.ridgedown)

minimo.l.ridgedown <- modelo.cv.ridgedown$lambda.min
minimo.l.ridgedown

#ridge con lambda mínimo
ridge.modelo.down <- glmnet(down_X,down_Y, family = "binomial", alpha=0, lambda = minimo.l.ridgedown, preProcess= c("center","scale"))
#coeficientes
ridge.modelo.down$beta

#predicciones
x.test.ridge.down <- model.matrix(P6920 ~ P6440 + P6780 + P6800 + P7040 + P7180 + INGLABO, data = testing)
ridge.down.predecido <- predict(ridge.modelo.down, newx =x.test.ridge.down, type="response")
ridge.down.predecido
predecir.ridge.down <- ifelse(ridge.down.predecido > 0.5, 1, 0)
predecir.ridge.down

#Performance del modelo
ridge.down_testing<-data.frame(testing,predecir.ridge.down)

ridge.down_testing<-rename(ridge.down_testing, prediccion_ridge.down =s0)
with(ridge.down_testing,prop.table(table(P6920,prediccion_ridge.down))) #9.75% son falsos negativos

pred_ridge.down<-prediction(as.numeric(predecir.ridge.down), as.numeric(ridge.down_testing$P6920))
roc_ridge.down <- performance(pred_ridge.down,"tpr","fpr")
plot(roc_ridge.down, main = "ROC curve", colorize = T)
abline(a = 0, b = 1)
auc_ridge.down <- performance(pred_ridge.down, measure = "auc")
auc_ridge.down@y.values[[1]] #AUC=0.767

confusionMatrix(as.factor(ridge.down_testing$prediccion_ridge.down),as.factor(ridge.down_testing$P6920)) #Accuracy: 0.862, sensitivity: 0.629, specificity: 0.905




