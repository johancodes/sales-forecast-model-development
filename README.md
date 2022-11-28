# sales-forecast-model-development

Imagine being able to remove the guesswork from sales forecasting. We used Deep Learning (LSTM, Keras TensorFlow) to train a model to do just that. Our model was trained on 7 years of monthly Norwegian car sales data. Please provide an input between 10000 and 14000 for months 1 through 5. Your monthly numbers can rise or fall over the 5 months. 

Notes: 

EXECUTIVE SUMMARY OF TIME SERIES FORECAST MODEL

We wanted to build a demo model that can forecast car sales in Norway. The model, after given sales for months 1 through 5 as input, should be able to forecast the 6th month's sales. 

A dataset of Norwegian monthly nationwide car sales from start of 2007 to end of 2016 - 10 years of sales data - was used in model development. 

We have successfully created a univariate (one type of input variable only i.e. sales per month), one-step (i.e. forecast the following month's sales if five preceeding months are given) time series forecasting model, with an LSTM deep learning algorithm. 

Average difference between forecasted and actual sales per month is about 987 units. Do take note that we are only using 85 rows as training data from the dataset. 

HOW ACCURATE IS OUR MODEL, BASED ON TEST DATA?

Average of differences between forecasted and actual sales for last 15 months of the dataset is 987 units. Average monthly sales for the same 15 months is 12780 units. 

As a description of accuracy, the percentage of difference average over actual average (987/12780*100) is 7.7%, which means forecasts are off by only 7.7% on average. When bearing in mind that our model was trained on only 85 rows of data, this is an acceptable accuracy.

From the plot below, we can see that our model is generally in step with the rising and falling of sales over a 15 month period.

MODEL DEVELOPMENT NOTES

Data must be in Series format before they can be passed into df_reshaped() which converts the data into the usable/required X and y format. X shape = (n, window_size, 1) and y shape = (n, 1).

Tensors are then feature scaled. Note that both X and y are scaled, not just X.
I do think we should feature scale the dataframe before we reshape them into usable tensors. Seems less complicated to do this.

When we want to pass in fresh, unseen data to obtain a forecast, we do not use the  df_reshaped() function to turn features into 3D tensors, we use numpy's .reshape(-1, 5, 1) -- where -1 indicates no change to SAMPLES, 5 is the window_size or TIMESTEPS of our model and 1 is the number of FEATURES we want to predict.

