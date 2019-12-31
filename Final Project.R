
#Installing packages
library(MASS)
library(lmtest)
library(stats)
library(dplyr)
library(ggplot2)
library(glmnet)
library(pander)
library(faraway)
library(ISLR)
library(gridExtra)
library(sqldf)
library(caret)
library(png)
library(grid)

## Introduction

## Data Description and Exploratory Analysis
### Variable Description

bike = read.csv("/Users/ddbor/Documents/MDA/STATS 9159A (Statistic Modelling I)/Assigments/Project/day.csv")

# The features instant, dteday, casual and cnt will be removed from the model 
bike = select (bike,-c(instant,dteday,casual,cnt))

# Column names
cname = names(bike)


# Correlation Matrix
pairs(~.,data=bike, main="Correlation Matrix")


# Making some plots for the data
m = which(bike$mnth<= 6)
monthdata = bike[m,]

ggplot(bike, aes(season, registered)) + geom_point(position = position_jitter(w = 1, h = 0), aes(color = atemp)) + scale_color_gradient(low = "#88d8b0", high = "#ff6f69") + labs(title="Registered vs Month", caption = "Figure 2", x = "Month", y = "Registered") + 
  theme(plot.title = element_text(hjust = 0.5))

bike$mnth = as.factor(bike$mnth)
bike$weathersit = as.factor(bike$weathersit)
bike$season = as.factor(bike$season)

weather_summary_by_season <- sqldf('select weathersit, season, avg(registered) as registered from bike group by weathersit, season')

ggplot(bike, aes(x=season, y=registered, color=weathersit))+geom_point(data = weather_summary_by_season, aes(group = weathersit))+geom_line(data = weather_summary_by_season, aes(group = weathersit))+ggtitle("Bikes Rent By Weather")+ scale_colour_hue('Weather',breaks = levels(bike$weathersit), labels=c('Good', 'Normal', 'Bad')) + labs(title="Registered vs Season", caption = "Figure 3", x = "Season", y = "Registered") + theme(plot.title = element_text(hjust = 0.5))
### Model Building 

# Let's separate our data in training and testing data and convert month and weathersit as factor. 
bike =select(bike, -c(temp, mnth, weekday))
#bike$mnth = as.factor(bike$mnth)
bike$weathersit = as.factor(bike$weathersit)
set.seed(0)
n = nrow(bike)
idx_tr <- sample(n,500,replace=FALSE)

biketr <- bike[idx_tr,]
bikets <- bike[-idx_tr,]


# Fitting Lasso
X_tr = model.matrix(registered~.,biketr)[, -1] #the first column (for intercept) is eliminated
y_tr = biketr$registered

X_ts = model.matrix(registered~.,bikets)[, -1] #the first column (for intercept) is eliminated
y_ts = bikets$registered

fit_lasso_cv = cv.glmnet(X_tr, y_tr, alpha = 1, lambda = exp(seq(from = -7, to = 7, by = 0.1)))


bestlam1 = fit_lasso_cv$lambda.min
fit_lasso_best = glmnet(X_tr, y_tr, alpha = 1, lambda = bestlam1)
df1 = as.data.frame(coef(fit_lasso_best)[,1], stringsAsFactors=FALSE)
names(df1) = "Coefficients"

#pander(df1)
plot(fit_lasso_cv)

title(sub ="Figure 4",
      cex.main = 2,   font.main= 4,
      cex.sub = 0.75, font.sub = 3,
      col.lab ="darkblue"
)

# Four models were fitted and the performance were evaluated using cross validation
# Fitted models:
# Model 1: Model with interactions and Quadratic Models
# Model 2: Model after performing lasso in Model 1
# Model 3: Model with interactions
# Model 4: Full Model 

set.seed(251082976)
train.control = trainControl(method="cv", number = 5)

# Model with original predictors plus only interactions between different predictors
model_interaction = train(registered~(season + yr + workingday + 
                                         holiday + weathersit + atemp + 
                                         hum + windspeed)^2, 
                           data=biketr, method="lm", trControl=train.control)

# Model that includes the original predictors, interactions and quadratic terms. 
model_quad = train(registered~(season+yr+holiday+workingday+weathersit+atemp+hum+windspeed)^2+
                     I(atemp^2)+I(hum^2)+I(windspeed^2), 
                   data=biketr, method="lm", trControl=train.control)

# Model after performing lasso in the quadratic model 
model_quad_lasso = train(registered~season+yr+holiday+workingday+weathersit+atemp+hum+windspeed+
  I(atemp^2) +I(hum^2)+season:yr+season:holiday + season:workingday +
  season:weathersit + season:atemp + season:hum + season:windspeed + 
  yr:holiday + yr:workingday + yr:weathersit + yr:atemp + holiday:weathersit + 
  workingday:weathersit + workingday:atemp + workingday:windspeed + weathersit:hum + weathersit:windspeed + hum:windspeed, data=biketr, method="lm", trControl=train.control)

# Model with the original predictors. 
fullmodelcv = train(registered~., 
                           data=biketr, method="lm", trControl=train.control)

model_quadratic = mean((model_quad$resample$RMSE)^2)
model_quadlasso = mean((model_quad_lasso$resample$RMSE)^2)
interaction_model = mean((model_interaction$resample$RMSE)^2)
fullmodelcvmse = mean((fullmodelcv$resample$RMSE)^2)

df2 = data.frame("Quadratic Model" = model_quadratic,
           "Quadratic Model (Lasso)"= model_quadlasso, 
           "Interaction Model" = interaction_model, 
           "Full Model" = fullmodelcvmse, check.names = FALSE)

pander(pandoc.table(df2, caption="MSE"))


# Lasso was performed to eliminate some variables from the quadractic model specified above. 

X_trq = model.matrix(registered~(season+yr+holiday+workingday+weathersit+atemp+hum+windspeed)^2+
                     I(atemp^2)+I(hum^2)+I(windspeed^2),
                     biketr)[, -1] #the first column (for intercept) is eliminated
y_trq = biketr$registered


fit_lasso_cvq = cv.glmnet(X_trq, y_trq, alpha = 1, lambda = exp(seq(from = -7, to = 7, by = 0.1)))


bestlam1q = fit_lasso_cv$lambda.min
fit_lasso_bestq = glmnet(X_tr, y_tr, alpha = 1, lambda = bestlam1)
df1q = as.data.frame(coef(fit_lasso_best)[,1], stringsAsFactors=FALSE)
names(df1q) = "Coefficients"

#pander(df1)
plot(fit_lasso_cvq)

title(sub ="Figure 4",
      cex.main = 2,   font.main= 4,
      cex.sub = 0.75, font.sub = 3,
      col.lab ="darkblue"
      )

# Fitting the full model using the training data with all the remaining variables

fullmodel = lm(registered~., data=biketr)
pander(summary(fullmodel), caption="Full Model")
#anova(fullmodel, model_int)

### Collinearity
# Calculating the VIF values 

pander(vif(fullmodel), caption="VIF")


### Model Assumptions
#### Equal Variance (EV) and Linearity. 

#Let's check the model assumptions
plot(resid(fullmodel)~fitted(fullmodel), col = "grey", pch = 20,
     xlab = "Fitted", ylab = "Residuals", main = "Fitted vs Residuals", sub="Figure 5",
     cex.lab=.8, cex.axis=.5, cex.main = 0.8, cex.sub =0.75)
abline(h = 0, col = "darkorange", lwd = 2)

bpval = bptest(fullmodel)$p.value

#### Normality 
#Let's check the model assumptions
qqnorm(resid(fullmodel), main = "Normal Q-Q Plot", col = "darkgrey",
       cex.lab=.8, cex.axis=.5, cex=0.8,  sub="Figure 6",
       cex.lab=.8, cex.axis=.5, cex.main = 0.8, cex.sub =0.75)
qqline(resid(fullmodel), col = "dodgerblue", lwd = 2)

spval = shapiro.test(resid(fullmodel))$p.value

#Let's check the model assumptions
par(mfrow=c(1,2), pin=c(3,2))
plot(resid(fullmodel)~fitted(fullmodel), col = "grey", pch = 20,
     xlab = "Fitted", ylab = "Residuals", main = "Fitted vs Residuals",
     cex.lab=.8, cex.axis=.8,)
abline(h = 0, col = "darkorange", lwd = 2)
abline(h = 0, col = "darkorange", lwd = 2)
qqnorm(resid(fullmodel), main = "Normal Q-Q Plot", col = "darkgrey",
       cex.lab=.8, cex.axis=.8,)
qqline(resid(fullmodel), col = "dodgerblue", lwd = 2)


# Test for the EV and Normality assumption 
# With this model only the linearity seems ok the qq plot shows heavy left tail... 

bpval = bptest(fullmodel)$p.value
spval = shapiro.test(resid(fullmodel))$p.value


#### Stabilizing the Model Assumptions

# Looking at the Box-Cox Curve to estimate the alpha and see if the transformation stabilizes the EV and Normality assumptions. 

boxcox(fullmodel,lambda = seq(0.7, 1, by = 0.005))
lambda = 0.85

# The cox-box transformation did not improved the model
# Hence this will be one of the limitations of the model

lm_cox <- lm(((registered^(lambda)-1)/(lambda)) ~., data = biketr)
par(mfrow=c(1,2), pin=c(3,2))
plot(resid(lm_cox)~fitted(lm_cox), col = "grey", pch = 20,
     xlab = "Fitted", ylab = "Residuals", main = "Fitted vs Residuals",
     cex.lab=.8, cex.axis=.8,)
abline(h = 0, col = "darkorange", lwd = 2)
abline(h = 0, col = "darkorange", lwd = 2)
qqnorm(resid(lm_cox), main = "Normal Q-Q Plot", col = "darkgrey",
       cex.lab=.8, cex.axis=.8,)
qqline(resid(lm_cox), col = "dodgerblue", lwd = 2)


# Model interactions
# The EV of this model looks better however the Normality Q-Q plot was still violated...
par(mfrow=c(1,2), pin=c(3,2))
plot(resid(model_interaction)~fitted(model_interaction), col = "grey", pch = 20,
     xlab = "Fitted", ylab = "Residuals", main = "Fitted vs Residuals",
     cex.lab=.8, cex.axis=.8,)
abline(h = 0, col = "darkorange", lwd = 2)
abline(h = 0, col = "darkorange", lwd = 2)
qqnorm(resid(model_interaction), main = "Normal Q-Q Plot", col = "darkgrey",
       cex.lab=.8, cex.axis=.8,)
qqline(resid(model_interaction), col = "dodgerblue", lwd = 2)

#bpinter = bptest(model_interaction)$p.value
spvalinter = shapiro.test(resid(model_interaction))$p.value


# Quadratic Model
# The EV and Normality were still violated... 
par(mfrow=c(1,2), pin=c(3,2))
plot(resid(model_quad)~fitted(model_quad), col = "grey", pch = 20,
     xlab = "Fitted", ylab = "Residuals", main = "Fitted vs Residuals",
     cex.lab=.8, cex.axis=.8,)
abline(h = 0, col = "darkorange", lwd = 2)
abline(h = 0, col = "darkorange", lwd = 2)
qqnorm(resid(model_quad), main = "Normal Q-Q Plot", col = "darkgrey",
       cex.lab=.8, cex.axis=.8,)
qqline(resid(model_quad), col = "dodgerblue", lwd = 2)

#bpinter = bptest(model_quad)$p.value
spvalinter = shapiro.test(resid(model_quad))$p.value

# Quadratic Model After Lasso
# The EV and Normality were still violated
par(mfrow=c(1,2), pin=c(3,2))
plot(resid(model_quad_lasso)~fitted(model_quad_lasso), col = "grey", pch = 20,
     xlab = "Fitted", ylab = "Residuals", main = "Fitted vs Residuals",
     cex.lab=.8, cex.axis=.8)
abline(h = 0, col = "darkorange", lwd = 2)
abline(h = 0, col = "darkorange", lwd = 2)
qqnorm(resid(model_quad_lasso), main = "Normal Q-Q Plot", col = "darkgrey",
       cex.lab=.8, cex.axis=.8,)
qqline(resid(model_quad_lasso), col = "dodgerblue", lwd = 2)

#bpinter = bptest(model_quad_lasso)$p.value
spvalinter = shapiro.test(resid(model_quad_lasso))$p.value


#### Influential Points
# Cook's distance

# The last obs is an influential point
inf = which(cooks.distance(fullmodel) > 4 / length(cooks.distance(fullmodel)))
con = cooks.distance(fullmodel) > 4 / length(cooks.distance(fullmodel))
par(mfrow=c(1,2), pin=c(3,2))
plot(resid(fullmodel)~fitted(fullmodel), col = factor(abs(rstandard(fullmodel)) > 2), pch = 20,
     xlab = "Fitted", ylab = "Residuals", main = "Outliers ",  sub="Figure 7",
     cex.lab=.8, cex.axis=.5, cex.main = 0.8, cex.sub =0.75)

abline(h = 0, col = "darkorange", lwd = 2)
title(sub ="Figure 7",
      cex.main = 2,   font.main= 4,
      cex.sub = 0.75,
      col.lab ="darkblue"
)
plot(resid(fullmodel)~fitted(fullmodel), col = factor(con), pch = 20,
     xlab = "Fitted", ylab = "Residuals", main = "Influential Points",  sub="Figure 8",
     cex.lab=.8, cex.axis=.5, cex.main = 0.8, cex.sub =0.75,
     colors = "blue")

abline(h = 0, col = "darkorange", lwd = 2)
title(sub ="Figure 8",
      cex.main = 2,   font.main= 4,
      cex.sub = 0.75,
      col.lab ="darkblue"
)

# bptest
bpval1 = bptest(fullmodel)$p.value

# Shapiro test
spval1 = shapiro.test(resid(fullmodel))$p.value

## Results
# Applying the model to the test data

pred_ls = predict(fullmodel, newdata=bikets) # prediction for test data
mse_ls = mean((pred_ls-bikets$registered)^2)

df3 = data.frame("Prediction" = pred_ls,
                 "Registered"= bikets$registered)

mae = mean(abs(pred_ls-bikets$registered))
average = mean(bikets$registered)


library(png)

img <- readPNG("/Users/ddbor/Documents/MDA/STATS 9159A (Statistic Modelling I)/Assigments/Project/lambda_path.png")
grid.raster(img)



