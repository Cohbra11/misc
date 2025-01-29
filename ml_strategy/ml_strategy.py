import numpy as np
import pandas as pd
import json
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
import backtrader as bt
import matplotlib.pyplot as plt  # Import Matplotlib

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_rsi(data, window):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_wt(data, n1, n2):
    hlc3 = (data['high'] + data['low'] + data['close']) / 3
    esa = hlc3.ewm(span=n1).mean()
    d = abs(hlc3 - esa).ewm(span=n1).mean()
    ci = (hlc3 - esa) / (0.015 * d)
    wt1 = ci.ewm(span=n2).mean()
    wt2 = wt1.rolling(window=4).mean()
    return wt1, wt2

def calculate_cci(data, window):
    tp = (data['high'] + data['low'] + data['close']) / 3
    ma = tp.rolling(window=window).mean()
    md = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    return (tp - ma) / (0.015 * md)

def calculate_adx(data, window):
    high = data['high']
    low = data['low']
    close = data['close']
    plus_dm = high.diff()
    minus_dm = low.diff()
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/window).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/window).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/window).mean()
    return adx

class MLStrategy(bt.Strategy):
    params = (
        ('model', None),
        ('scaler', None),
        ('features', None),
    )

    def __init__(self):
        self.model = self.params.model
        self.scaler = self.params.scaler
        self.features = self.params.features
        self.data_features = {feature: self.datas[0].__getattr__(feature) for feature in self.features}

    def next(self):
        # Prepare the feature data for prediction
        feature_values = np.array([self.data_features[feature][0] for feature in self.features]).reshape(1, -1)
        feature_df = pd.DataFrame(feature_values, columns=self.features)
        logging.debug("Feature DataFrame for prediction: %s", feature_df)
        normalized_features = self.scaler.transform(feature_df)
        normalized_features_df = pd.DataFrame(normalized_features, columns=self.features)
        logging.debug("Normalized features for prediction: %s", normalized_features_df)
        
        # Predict the signal (1 for buy, 0 for hold/sell)
        signal = self.model.predict(normalized_features_df)[0]
        logging.debug("Predicted signal: %d", signal)
        
        if signal == 1 and not self.position:
            self.buy()
        elif signal == 0 and self.position:
            self.sell()

class CustomPandasData(bt.feeds.PandasData):
    lines = ('hlc3', 'RSI_f1', 'WT1_f2', 'WT2_f2', 'CCI_f3', 'ADX_f4', 'RSI_f5')
    params = (
        ('hlc3', -1),
        ('RSI_f1', -1),
        ('WT1_f2', -1),
        ('WT2_f2', -1),
        ('CCI_f3', -1),
        ('ADX_f4', -1),
        ('RSI_f5', -1),
    )

class MLBacktest:
    def __init__(self, data, settings):
        self.data = data
        self.settings = self.Settings(settings)
        self.scaler = MinMaxScaler()
        self.normalized_data = None
        self.model = None

        logging.debug("MLBacktest initialized with settings: %s", settings)

    class Settings:
        def __init__(self, settings):
            self.source = settings['source']
            self.features = {f"{settings['features'][key]['type']}_{key}": settings['features'][key] for key in settings['features']}

    def calculate_features(self):
        logging.debug("Calculating features...")
        for key, feature in self.settings.features.items():
            feature_type, feature_key = key.split('_')
            paramA = feature['paramA']
            paramB = feature.get('paramB', None)
            if feature_type == 'RSI':
                self.data[key] = calculate_rsi(self.data, paramA)
            elif feature_type == 'WT':
                wt1, wt2 = calculate_wt(self.data, paramA, paramB)
                self.data[f'WT1_{feature_key}'] = wt1
                self.data[f'WT2_{feature_key}'] = wt2
            elif feature_type == 'CCI':
                self.data[key] = calculate_cci(self.data, paramA)
            elif feature_type == 'ADX':
                self.data[key] = calculate_adx(self.data, paramA)
        logging.debug("Features calculated: %s", self.data.head())

    def normalize_features(self):
        logging.debug("Normalizing features...")
        # Extract features to be normalized
        feature_columns = ['open', 'high', 'low', 'close']
        for key, feature in self.settings.features.items():
            if feature['type'] == 'WT':
                feature_columns.append(f'WT1_{key}')
                feature_columns.append(f'WT2_{key}')
            else:
                feature_columns.append(f"{feature['type']}_{key}")
        features = self.data[feature_columns]
        logging.debug("Original features: %s", features.head())

        # Handle NaN values by filling them with the mean of the column
        features = features.fillna(features.mean())

        # Normalize features using MinMaxScaler
        self.normalized_data = pd.DataFrame(self.scaler.fit_transform(features), columns=feature_columns)
        logging.debug("Normalized features: %s", self.normalized_data.head())

    def convert_to_classes(self, y):
        # Ensure y is a pandas Series
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        # Convert continuous target values into discrete classes
        y_class = np.where(y.diff() > 0, 1, 0)  # 1 for up, 0 for down or neutral
        y_class = pd.Series(y_class).shift(-1)  # Shift to align with features
        y_class = y_class.dropna()  # Remove NaN values
        return y_class

    def train_model(self):
        logging.debug("Training model...")
        # Prepare the data for training
        X = self.normalized_data
        y = self.data[self.settings.source]
        y_class = self.convert_to_classes(y)
        X = X.iloc[:len(y_class)]  # Align X and y_class

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
        logging.debug("Training features: %s", pd.DataFrame(X_train, columns=self.settings.features).head())
        logging.debug("Training labels: %s", pd.Series(y_train).head())

        # Initialize the Random Forest model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Initialize the ensemble model using VotingClassifier
        ensemble_model = VotingClassifier(estimators=[('rf', rf_model)], voting='soft')

        # Train the model
        ensemble_model.fit(pd.DataFrame(X_train, columns=self.settings.features), y_train)
        self.model = ensemble_model

        # Log the feature names used by the RandomForestClassifier
        if hasattr(rf_model, 'get_feature_names_out'):
            logging.debug("RandomForestClassifier feature names: %s", rf_model.get_feature_names_out())
        else:
            logging.debug("RandomForestClassifier feature names: %s", self.settings.features)

        # Evaluate the model
        y_pred = ensemble_model.predict(pd.DataFrame(X_test, columns=self.settings.features))
        accuracy = accuracy_score(y_test, y_pred)
        logging.info("Model accuracy: %f", accuracy)

    def run_backtest(self):
        logging.debug("Running backtest...")
        # Calculate features
        self.calculate_features()

        # Normalize features
        self.normalize_features()

        # Train the model
        self.train_model()

        # Prepare the data for backtrader
        self.data['datetime'] = pd.to_datetime(self.data['time'], unit='ms')
        self.data.set_index('datetime', inplace=True)
        data = CustomPandasData(dataname=self.data)

        # Initialize cerebro
        cerebro = bt.Cerebro()
        cerebro.adddata(data)
        cerebro.addstrategy(MLStrategy, model=self.model, scaler=self.scaler, features=list(self.settings.features.keys()))

        # Set initial cash
        cerebro.broker.set_cash(10000)

        # Run the backtest
        cerebro.run()

        # Plot the results
        cerebro.plot()

# Example usage
if __name__ == "__main__":
    try:
        logging.debug("Starting backtest...")
        # Load data from JSON file
        with open('../backtesting_data.json', 'r') as data_file:
            data = pd.read_json(data_file)

        # Add 'hlc3' column for the weighted average of high, low, and close prices
        data['hlc3'] = (data['high'] + data['low'] + data['close']) / 3

        # Load settings from the JSON file
        with open('settings.json', 'r') as settings_file:
            settings = json.load(settings_file)

        # Initialize and run the backtest
        backtest = MLBacktest(data, settings)
        backtest.run_backtest()

    except Exception as e:
        logging.error("An error occurred: %s", e)

    # Keep the console open
    input("Press Enter to exit...")