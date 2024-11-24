# DeepLearningCryptoPrediction

Deep learning (LSTM) to predict crypto price prediction using Recursive Forecasting

Uses 'Close', 'Open', 'High', 'Low', and 'Volume' as features.

The following graphs are tested on Bitcoin.

![Screenshot 2024-11-24 at 5 24 16 PM](https://github.com/user-attachments/assets/247fa3ef-bae4-43cf-ac28-2153d2af333e)

![Screenshot 2024-11-24 at 5 24 35 PM](https://github.com/user-attachments/assets/8d86094d-f32a-4b8e-adf1-696af7df7724)

This project is ongoing, there is still a lot of hyperparameter tuning needed to be done.

## Limitations:
* Recursive Forecasting has inherent limitations. For example, the error accumulates so the predictions can only be accurate in the short term.
* The predictions only use the features described above, whereas true cryptocurrency prices are dictated by other measures, such as public sentiment, macroeconomic indicators etc.
* The cryptocurrency market is known for its high volatility and rapid changes. This unpredictability can make it challenging for any model to consistently provide accurate forecasts, especially when relying on historical data alone.
