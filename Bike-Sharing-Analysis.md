Bike Sharing Analysis
================
Daniela Borges
31/12/2019

## Introduction

Bike-sharing systems transform the traditional bike rental process where
obtaining membership, rental and bike returned have been automated via a
network of bike stations. These systems allowed the riders to take the
bike at any bike station and returned it to the closest one to their
destination. Today, there exists plenty of interest in these systems
because most cities are highly populated, and the implementation of
these benefits the traffic, environmental, and health issues.

The objective of the project was to perform a multiple linear regression
analysis to predict the number of registered members that used this
bike-sharing system, and identify behaviors depending on the
environmental and seasonal settings. For instance, weather conditions,
precipitation, and other factors could affect rental behavior.
Performing this analysis would help to understand better the trends
depending on the season and the relationship between the total number of
registered users and the variables of interests.

## Data Description and Exploratory Analysis

### Variable Description

The data of the project was provided by Capital Bikeshare, a
subscription-based bike-sharing company, located in Washington DC from
2011 to 2012. This data consisted of weather condition factors and if
the day of registration was a working day or not. Therefore, the
variables involved in the data were the following.

  - **Instant**. Record Index
  - **Date**. Date when the rider rented the bike
  - **Season**. Season of the year when the bike was rented
      - Spring, summer, fall, and winter
  - **Year**. Year when the survey was done
  - **Month**. The month when the bike was rented, from January to
    Decemeber
  - **Holiday**. Specifies whether the current date when the bike was
    rented was a holiday or not
  - **Week day**. Day of the week the bike was rented, from Sunday to
    Saturday
  - **Working day**. Specifies whether the bike was rented on a working
    day or not
  - **Weather sit**. Characteristics of the weather when the bike was
    rented
      - Clear, few clouds or partly cloudy
      - Mist and cloudy; mist and broken clouds; mist and few clouds;
        mist
      - Light snow; light rain with thunderstorm and scattered clouds;
        light rain and scattered clouds
      - Heavy rain with ice pallets thunderstorm and mist; snow and fog
  - **Temperature**. Normalized temperature in Celsius in hourly scale
    when the bike was rented.
  - **Feels like temperature**. The normalized feeling temperature in
    Celsius on hourly scale when the bike was rented.
  - **Humidity**. Normalized humidity when the bike was rented.
  - **Wind speed**. Normalized wind speed when the bike was rented.
  - **Registered**. Total of riders that rented a bike which membership
    hast the following characteristics: annual member, 30-day member or
    day key member

| Variable               | Name       | Values         | Type        |
| ---------------------- | ---------- | -------------- | ----------- |
| Instant                | instant    | \[1, .., 731\] | int         |
| Date                   | dteday     | dd/mm/yyyy     | object      |
| Season                 | season     | \[1, 2, 3, 4\] | categorical |
| Year                   | yr         | \[2011, 2012\] | binary      |
| Month                  | mnth       | \[1, …,12\]    | categorical |
| Holiday                | holiday    | \[0,1\]        | binary      |
| Week day               | weekday    | \[0, …, 6\]    | categorical |
| Working day            | workingday | \[0,1\]        | binary      |
| Weather sit            | weathersit | \[1,2,3\]      | categorical |
| Temperature            | temp       | (0,1)          | float       |
| Feels like temperature | atemp      | (0,1)          | float       |
| Humidity               | hum        | (0,1)          | float       |
| Wind speed             | windspeed  | (0,1)          | float       |
| Registered             | registered | \[20, …, 246\] | int         |

After performing a descriptive analysis of the data, it was decided that
the instant and date were not necessary as predictors.The reason for
this was due to that instant was just a record and other variables could
represent the date. Futhermore, the continuous variables were already
normalized so there was no need to do it to perform the modelling.

In this step, a correlation matrix would be used to understand the
variables and assessed the existing relationship between them with the
help of a correlation matrix (Figure 1).

Relevant outcomes from the correlation matrix were as follows:

  - atemp and temp variables are highly correlated, meaning that one can
    be explained by the other. Therefore, only the most correlated with
    the objective variable would be kept
  - season and month show a linear relationship, which means that they
    are highly correlated, and we need remove one of them later
  - The scatter plot between registered and weekday does not show
    significance change, so weekday should be removed
  - Feels like temperature and month have a relationship due to the
    number of users registered behavior can change depending on the
    month and the temperature

![](Bike-Sharing-Analysis_files/figure-gfm/setup%20q1.3-1.png)<!-- -->

Now, the relationship between the registered users and the feels like
temperature during each season (Figure 2) shows that users’ registration
tends to peak during the summer and decreases during the winter.

<img src="Bike-Sharing-Analysis_files/figure-gfm/setup q1.4-1.png" style="display: block; margin: auto;" />

Additionally, Figure 3 shows that people register more when the weather
conditions are better which also reafirms that during the winter season
this number tends to decrease.
<img src="Bike-Sharing-Analysis_files/figure-gfm/setup q1.5-1.png" style="display: block; margin: auto;" />

The variables that were removed before modelling were “temperature”,
“month” and “weekday”, they could lead to multicollinearity and
innacurate estimations. It was decided to keep “season” in the model
instead of “month”, because they had high correlation and season
explained more the objective variable and had smaller number of
categries in it. Thus, because month was removed, including a quadratic
term might no be necessary because it is the only one that show a curvy
pattern with some of the other variables. As for “weekday”, its
scatterplot between the objective variable did not show signigicant
changes and could be represented with the working day variable.

## Model Description

### Methodology

First, the data was splited into around 70% - 30% between training and
testing stages. The model learned on the training data in order to be
generalized in the test set. Second, a linear regression model using
lasso as penalization was used to give an idea of which predictors were
important and which could be removed. Afterward, different models were
fitted, and cross-validation was used to evaluate its performance using
MSE as criteria. Subsequently, the assumptions of the model selected in
the last step were verified; these were linearity, equal variance, and
normality. In this step, residual tests and diagnostic plots were
performed. Also, influential points and outliers were analyzed. Finally,
model validation was used on the test set, using the model with the best
performance

### Model Building

Lasso regression was performed in the full first-order model, and it was
found that all the coefficients were necessary for the model (different
than zero). Furthermore, the results indicated that penalizing the model
increased the mean squared error (Figure 4). Hence, the first-order
model with no penalization was preferable.

<img src="Bike-Sharing-Analysis_files/figure-gfm/set up q1.7-1.png" style="display: block; margin: auto;" />

After performing lasso, it was decided to explore three models besides
the original one, these three included interactions because one of the
objectives was to explain how the relationships between weather
conditions and seasons with the other factors affected the total number
of users’ registrations. The models were explained in Table 2.

| Model                   | Characteristics                                                                                                  |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Quadratic Model         | This model included all the quadratic terms of each predictors and all the interactions between each one of them |
| Quadratic Model (Lasso) | This model included all the variables with non zero coefficients after performing lasso in the Quadractic Model  |
| Interaction Model       | This model included only the original predictors plus interactions between different predictors                  |
| Full model              | This model included the original predictors                                                                      |

Models Fitted

Cross-validation was used to evaluate the performance of each model
using 5 folds and Root Mean Square Error (MSE) as criteria (Table 3).
The results indicated that the best model was the OLS model because it
had the smaller MSE.

| Quadratic Model | Quadratic Model (Lasso) | Interaction Model | Full Model |
| :-------------: | :---------------------: | :---------------: | :--------: |
|     601299      |         700864          |      3424131      |   377431   |

MSE

The results of the previous analysis showed that the simple linear
regression model had the best performance overall. Because of this, it
was chosen to model the data, and would be used to make future
predictions.

### Model Fitting

After fitting the full model, the majority of the predictors were
significant (Table 5); this is because their p-values were less than the
significance level of 5%. Furthermore, holiday had a p-value higher than
the significance level; however, it was left in the model because its
value was not significantly higher. The \(Adjusted R^2 = 0.8513\)
indicated that this model explained most of the variability of the data
considering the number of predictors.

From the coefficients, it was expected that people tended to use the
service less during adverse weather conditions(thunderstorm, light rain,
or strong wind speed) or holidays, because of the negative coefficients.
These means that each unit change in these variables would decrease the
number of registrations.

|                 | Estimate | Std. Error | t value | Pr(\>|t|) |
| :-------------: | :------: | :--------: | :-----: | :-------: |
| **(Intercept)** |  629.5   |   196.9    |  3.197  | 0.001479  |
|   **season2**   |   861    |   100.1    |  8.605  | 1.056e-16 |
|   **season3**   |  859.3   |   130.1    |  6.602  | 1.058e-10 |
|   **season4**   |   1377   |   87.07    |  15.81  | 9.006e-46 |
|     **yr**      |   1788   |   54.67    |  32.7   | 4.74e-125 |
|   **holiday**   | \-300.6  |   187.5    | \-1.604 |  0.1095   |
| **workingday**  |   1019   |   59.17    |  17.22  | 2.992e-52 |
| **weathersit2** | \-364.5  |   71.18    | \-5.121 | 4.387e-07 |
| **weathersit3** |  \-1728  |   185.8    | \-9.302 | 4.638e-19 |
|    **atemp**    |   3358   |   307.9    |  10.91  | 6.134e-25 |
|     **hum**     | \-689.9  |   262.2    | \-2.631 | 0.008781  |
|  **windspeed**  |  \-1572  |   372.9    | \-4.215 | 2.976e-05 |

| Observations | Residual Std. Error | \(R^2\) | Adjusted \(R^2\) |
| :----------: | :-----------------: | :-----: | :--------------: |
|     500      |        599.5        | 0.8546  |      0.8513      |

Full Model

### Collinearity

The collinearity of the model was verified by lookig at the variance
inflation value (VIF). Based on a VIF cut off equal to 10, because none
of the predictors had a higher VIF than the cut off, there was no severe
collinear issue between the predictors, hence the variance of the
estimates were not inflated, hence the p-values can be trusted and that
means that the true values of the predictors were closer to the
estimated coefficients.

### Model Assumptions

#### Equal Variance (EV) and Linearity.

On one side, the EV of the model was violated. The residual plot showed
that the distribution belt of points became wider along with x-axis.
Also, the Breusch-Pagan test p-value was smaller than the significance
level, which confirmed the violation of the EV. On the other side, the
linearity assumptions was met, the residual plot did not show a pattern
different than linear.

<img src="Bike-Sharing-Analysis_files/figure-gfm/set up q1. 12-1.png" style="display: block; margin: auto;" />

| Test          | P-value         |
| ------------- | --------------- |
| Breusch-Pagan | 4.86569910^{-6} |

Equal Variance Test

#### Normality

The normality assumption was verified using a Q-Q plot and the Shapiro
Wilk test. The Q-Q plot revealed that the distribution had a heavy left
tail, and the p-value of the test was lower than the significance level.
Both of these confirmed the violation of the normality assumption.

<img src="Bike-Sharing-Analysis_files/figure-gfm/set up q1. 13-1.png" style="display: block; margin: auto;" />

| Test         | P-value          |
| ------------ | ---------------- |
| Shapiro-Wilk | 5.578863910^{-9} |

Normality Test

#### Stabilizing the Model Assumptions

The Box-Cox Transformation was used to stabilized the E.V. and
Normality, the lambda used to transformed the registered variable was
0.85, which was the most close to the real value of the lambda.
Neverthless, the assumptions were still violated.

#### Influential Points

The outliers shown in Figure 5 were detected by calculating the absolute
value of the standardized residuals greater than two, and the
influential points were calculated using Cook’s distance. If the
influential points were considered as errors and were removed from the
model, the EV and normality assumptions were still violated. However, by
analyzing the data, there was no reason for them to be removed, and they
did not look severe to change the model significantly, hence they were
kept.

![](Bike-Sharing-Analysis_files/figure-gfm/set%20up%20q1.%2017-1.png)<!-- -->

## Results

The model was applied to the test set and the resulted MSE was 44%
higher compared to the training set, thus it could be infered that the
selected model was overfitting. Looking at the mean absolute error (MAE)
averagely our model predictions were off by approximately 522 registered
users in comparison to the average of the test data. Hence, it can be
concluded that the predictions were very accurate, because they were
close to the actual values.

| MSE            | MAE     | \(\overline{Y}\) |
| -------------- | ------- | ---------------- |
| 5.47195610^{5} | 522.379 | 3549.502         |

Performance Metrics

The coefficients included in the final model were year, the feels like
temperature, working day, season, holiday, weather sit, humidity and
wind speed. Looking at the coefficient path the most important
predictors of this model were season, year, working day and weather sit
because they did not converge to zero as fast as the others.

![](Bike-Sharing-Analysis_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

## Conclusions

The goal of this project was to fit a multiple linear regression that
could be suitable to predict the number of registered users. To get to
the results, first, data preprocessing was performed to identify the
existing relationship between the variables, where it was discovered
some of the predictors were highly correlated. By observing in more
detail some plots between the objective variable and some predictors, it
was expected that the number of registered users behaved differently
according to the season and weather conditions. Before building the
model, lasso regression was used to check the significance for each
predictor. This led to retain all of the variables and by observing the
behavior of the regularization penalty (Figure 4), a model with no
penalization would perform better than a penalized model. Because of
this, four models were fitted (Table 2) and cross-validation was used to
look at the performance. It turned out that the regression model with no
interactions or quadratics terms reduced the training error (MSE). The
other models were decreasing the performance, which means they were
overfitting

After deciding in a model-based in its performance, the collinearity,
model assumptions, and influential points were checked. First, the
variance inflation value revealed that there was no severe collinearity
between the predictor variables. In terms of model assumptions,
unfortunately, the model violated the equal variance and normality, by
transforming the objective variable using the help of Box-Cox
transformation. Despite that, these transformations could not fix the
assumptions either adding quadratic terms nor interactions.
Subsequently, the influential points and outliers were checked (Figure
7-8). It could not be proved that there were measure errors so they
could not be casually removed. In fact, even if they were removed, the
model assumptions still were not met.

Finally, looking at the MSE of the test set, it could be concluded that
using multiple linear regression was not suitable for the research data.
The model chosen had the less mean squared error, but the model
assumptions were not met. In addition, because the MAE was small
considering the average number of registered users, the predictions
using this model were accurate enough.
