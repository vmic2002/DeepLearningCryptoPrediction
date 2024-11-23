# DeepLearningCryptoPrediction

Deep learning (LSTM) to predict crypto price prediction using Recursive Forecasting

Uses 'Close', 'Open', 'High', 'Low', and 'Volume' as features.

The following graphs are tested on Bitcoin.

![Screenshot 2024-11-23 at 6 14 15 PM](https://github.com/user-attachments/assets/dadb5537-db22-40f7-91ad-0d2d70bbf7b2)

![Screenshot 2024-11-23 at 6 14 49 PM](https://github.com/user-attachments/assets/f10fab71-7e3c-4818-bb8e-b26f129bcbba)

This project is ongoing, there is still a lot of hyperparameter tuning needed to be done.

## Limitations:
* Recursive Forecasting has inherent limitations. For example, the error accumulates so the predictions can only be accurate in the short term.
* The predictions only use the features described above, whereas true cryptocurrency prices are dictated by other measures, such as public sentiment, macroeconomic indicators etc.
* The cryptocurrency market is known for its high volatility and rapid changes. This unpredictability can make it challenging for any model to consistently provide accurate forecasts, especially when relying on historical data alone.
