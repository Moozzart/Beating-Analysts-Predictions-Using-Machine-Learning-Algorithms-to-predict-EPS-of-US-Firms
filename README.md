# Beating-Analysts-Predictions-Using-Machine-Learning-Algorithms-to-predict-EPS-of-US-Firms
## Task
To develop a model to predict the earnings per share of firms using macroeconomic and company fundamentals. And to test its predictive power against the analyst’s predictions.

## Data Used 
Extracted from IBES. Macroeconomic variables were taken from FRED. And since the data used is really large, I'm attaching my drive link of the data here: https://drive.google.com/drive/folders/1dxpd89Pqjwy1U_upZvYjZR7vNejEKvLf?usp=sharing

## More About The Project
Analysts from Wall Street usually do a great amount of research of different companies’ stocks, their past, the news related to those companies and their sentiments related to the company. They spend years getting to know the industry well and they are very up-to-speed about product cycles, competitive dynamics, perceived quality of management, etc

But doing all this can we say that analysts precisely tell you about the stocks? Sometimes and sometimes not although most of the times they are close to the actual stat. The problem here however is that they are all human and it is natural to have biases in one form or the other in the predictions.

We, with machine learning and deep learning techniques, tried to predict the EPS of the firms in the US  by using company fundamentals, macro-economic factors and indicators in which most of it were leading economic indicators.

## Task/Goal
-To use the macroeconomic factors and past quarters’ EPS to predict to present quarters’ EPS

-To use the company fundamentals and past quarters’ EPS to predict present quarters’ EPS

-To find the correlation between our predictions and with analysts predictions and the actual EPS to see whether our predictions follow the same trend as the actual EPS and analysts’ EPS predictions

-To find the number of times that we beat the analysts and with respect to every company in a given time period, what was our rank with respect to other analysts who predicted for the same company 

## Challenges
-The biggest challenge for us was to arrange the data in a specified order for it to feed them in the model because by using macro economic data and EPS, we had to use the same macro economic factors for all the firms in a given period of time but when we used financials and company fundamentals, we had different data for different firm

-The second challenge that we faced was that the ticker symbol with which the firms are identified once gets used by other firms in the future due to closure of that former firm. So, we finally used IBES ticker symbols for each of our firms, since they are unique to all the firms.

-The challenges in modelling were the multicollinearity problem, using severe garbage features, and not feeding any tint of the present quarters data. In this regard we even gave the advantage to the analysts by using some of the data of the last quarter as a proxy for the present quarter to predict the EPS because some of the data releases in between the quarters or on some odd date.

## Implementation
-In the first task we used macroeconomic factors, EPS of the last quarter ,EPS of same quarter of last year, yoy EPS change of last quarter and shock(defined by the difference between the EPS at time t-1 with average of EPS of t-2, t-3 and t-4, here t is the present quarter). **Let’s call the data without macroeconomic data as EPS_data.

-In the second task we included company fundamentals and EPS_data 

-In the third task we just included the EPS_data but without qoq EPS change.

-We used random forest regressor and ANN modelling to predict the actual EPS from the adequate features

# Results
-After merging with individual analysts estimates of past periods, the average percentile beat is 28.829 percentile

-Before merging with individual analysts estimates of past periods, the average percentile beat is 27.387 percentile 
