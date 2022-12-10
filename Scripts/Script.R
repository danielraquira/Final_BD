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


confusionMatrix(as.factor(lasso_testing$prediccion_lasso),as.factor(lasso_testing$P6920))