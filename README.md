# Diamond-price-prediction
Diamond Price Prediction using LSTM
This project builds a Long Short-Term Memory (LSTM) neural network to predict diamond market prices based on historical time series data.
It uses a deep learning sequence model to capture trends and forecast future prices.

 Features
Predicts future diamond prices using an LSTM-based neural network trained on historical closing prices.
Implements time series forecasting with a 60-day sliding window.

Visualizes:

Training vs validation data

Predicted vs actual prices

 Evaluates model performance using:

Root Mean Squared Error (RMSE)

Mean Absolute Error (MAE)

 Compares two LSTM architectures:

A deeper LSTM with 100 neurons

A simpler LSTM with 25 neurons

 How it works
 Data loading & preprocessing
Loads historical diamond market data from a CSV file downloaded from Yahoo Finance.

Uses the 'Close' price as the primary feature for prediction.

Normalizes the data to a 0-1 range using MinMaxScaler to help the neural network converge.

 Sequence creation
Creates input sequences of the past 60 days to predict the next day’s price.

Splits data into 80% training / 20% testing.

LSTM model
Defines an LSTM model in TensorFlow Keras with:

Two stacked LSTM layers

Dense layers for final regression output

Compiles with adam optimizer and mean_squared_error loss.

 Model evaluation
Uses RMSE and MAE to quantify prediction error.

Inversely transforms predictions to original scale (USD).

 Visualization
Plots predicted prices vs actual prices on a time series chart.


 Key algorithms & libraries
TensorFlow / Keras: LSTM model building

scikit-learn: MinMax scaling, MAE calculation

Matplotlib & Seaborn: Visualization

NumPy & Pandas: Data manipulation

 How to run
1️ Clone this repository:

bash
Copy
Edit
git clone https://github.com/KgosiSino/diamond-lstm.git
cd diamond-lstm
2️ Install dependencies:

bash
Copy
Edit
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn python-dateutil
3️ Run the notebook or script:

bash
Copy
Edit
jupyter notebook diamond_lstm_prediction.ipynb
# OR
python diamond_lstm_model.py
Model architecture
 First model
2 stacked LSTM layers with 100 neurons each

Dense layers with 25 and 1 neuron(s)

Trained for 100 epochs with batch size 32.

 Second model
2 stacked LSTM layers with 25 neurons each

Same Dense layers

Trained for 25 epochs with batch size 10.

 Example results
RMSE on test data: typically low, indicating good fit.
MAE also reported to give clear interpretation of average error in USD.



License
This project is licensed under the MIT License.

Acknowledgments
Dataset sourced from Yahoo Finance.

Built using TensorFlow & scikit-learn.

