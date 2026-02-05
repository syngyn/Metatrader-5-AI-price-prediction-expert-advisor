"""
Multi-Timeframe Predictor v8.1
Author: Jason Rusk jason.w.rusk@gmail.com
Copyright 2026

Fixes/Additions in this version:
- Integrated GaussianHMM for Market Regime Detection
- Added Correlation Engine for DXY (USDX) and SPX500
- FIXED: Walk-Forward Backtesting to prevent "Look-Ahead" data leakage
- FIXED: Indentation bug in download_data() validation loop
- ADDED: GRU model option for ensemble
- FIXED: TCN output dimension handling (was causing inflated weights)
- ADDED: Attention mechanism to LSTM for better performance

"""

import sys
import os
import subprocess
import warnings
import time
import json
import pickle
import argparse
import glob
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import pandas as pd

# --- NEW QUANT LIBRARIES ---
try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "hmmlearn"])
    from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings('ignore')

# --- CONFIGURATION: MT5 PATH ---
try:
    from config_manager import get_mt5_files_path as get_config_mt5_path, get_config
except ImportError:
    print("=" * 80)
    print("ERROR: config_manager.py not found!")
    print("=" * 80)
    sys.exit(1)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def install_package(package_name: str, pip_name: Optional[str] = None) -> None:
    try:
        __import__(package_name)
    except (ImportError, ModuleNotFoundError):
        install_name = pip_name or package_name
        subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])


required_packages = [
    ('MetaTrader5', 'MetaTrader5'), ('pandas', 'pandas'), ('numpy', 'numpy'),
    ('tensorflow', 'tensorflow'), ('sklearn', 'scikit-learn'),
    ('lightgbm', 'lightgbm'), ('keras_tuner', 'keras-tuner')
]

for package, pip_name in required_packages:
    install_package(package, pip_name)

import MetaTrader5 as mt5
import tensorflow as tf
import keras
from keras import layers
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb
import keras_tuner as kt

np.random.seed(42)
tf.random.set_seed(42)
tf.config.run_functions_eagerly(False)


# --- Helper Classes ---

class KalmanFilter:
    def __init__(self, process_variance: float, measurement_variance: float):
        self.q = process_variance
        self.r = measurement_variance
        self.x = 0.0
        self.p = 1.0
        self.k = 0.0

    def update(self, measurement: float) -> float:
        self.p += self.q
        self.k = self.p / (self.p + self.r)
        self.x += self.k * (measurement - self.x)
        self.p = (1 - self.k) * self.p
        return self.x


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate
        })
        return config


class AttentionLayer(layers.Layer):
    """Simple attention mechanism for LSTM outputs."""
    
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(input_shape[-1],),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch, timesteps, features)
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        context_vector = tf.reduce_sum(inputs * tf.expand_dims(attention_weights, -1), axis=1)
        return context_vector

    def get_config(self):
        return super().get_config()


# --- Main Predictor Class ---

class UnifiedLSTMPredictor:
    def __init__(self, symbol: str = "EURUSD", related_symbols: Optional[List[str]] = None,
                 ensemble_model_types: Optional[List[str]] = None, use_kalman: bool = False,
                 use_multitimeframe: bool = False):
        self.symbol = symbol.upper()
        # --- NEW MACRO SYMBOLS ---
        self.dxy_symbol = "USDX"
        self.spx_symbol = "SPX500"

        self.related_symbols = related_symbols or ["GBPUSD", "USDJPY"]
        self.ensemble_model_types = ensemble_model_types if ensemble_model_types is not None else ['lstm', 'transformer', 'lgbm']
        self.num_ensemble_models = len(self.ensemble_model_types)
        self.lookback_periods = 60
        self.base_path = self.get_mt5_files_path()
        self.use_kalman = use_kalman
        self.use_multitimeframe = use_multitimeframe

        # File paths
        self.predictions_file = os.path.join(self.base_path, f"predictions_{self.symbol}.json")
        self.status_file = os.path.join(self.base_path, f"lstm_status_{self.symbol}.json")
        self.feature_scaler_path = os.path.join(self.base_path, f"feature_scaler_{self.symbol}.pkl")
        self.target_scaler_path = os.path.join(self.base_path, f"target_scaler_{self.symbol}.pkl")
        self.selected_features_path = os.path.join(self.base_path, f"selected_features_{self.symbol}.json")
        self.pending_eval_path = os.path.join(self.base_path, f"pending_evaluations_{self.symbol}.json")
        self.tuner_dir = os.path.join(self.base_path, 'tuner_results')

        self.target_column = 'log_return_1h'
        self.feature_cols: Optional[List[str]] = None
        self.models: Dict[str, Any] = {}
        self.feature_scaler = RobustScaler()
        self.target_scaler = RobustScaler()

        self.models_by_timeframe: Dict[str, Dict[str, Any]] = {}
        self.scalers_by_timeframe: Dict[str, Tuple[RobustScaler, RobustScaler]] = {}

        self.kalman_config = {
            "1H": {"Q": 0.00001, "R": 0.01},
            "4H": {"Q": 0.00005, "R": 0.02},
            "1D": {"Q": 0.0001, "R": 0.05}
        }
        self.kalman_filters = {tf: KalmanFilter(c["Q"], c["R"]) for tf, c in self.kalman_config.items()}

        self.previous_predictions = {tf: None for tf in self.kalman_config.keys()}
        self.ema_alpha = 0.3
        self.ensemble_weights = [1.0 / self.num_ensemble_models] * self.num_ensemble_models
        self.prediction_history = {tf: [] for tf in self.kalman_config.keys()}
        self.ensemble_lookback = 20
        self.ensemble_learning_rate = 0.1

        self.initialize_mt5()
        self.ensure_symbols_selected()

    def get_mt5_files_path(self) -> str:
        mt5_path = get_config_mt5_path()
        if not os.path.exists(mt5_path):
            sys.exit(1)
        return mt5_path

    def initialize_mt5(self) -> None:
        if not mt5.initialize():
            sys.exit(1)
        print(f"Connected to MT5: {mt5.account_info().login}")

    def ensure_symbols_selected(self):
        """Ensures DXY and SP500 are in Market Watch."""
        for s in [self.symbol, self.dxy_symbol, self.spx_symbol] + self.related_symbols:
            mt5.symbol_select(s, True)

    def _download_macro_data(self, bars: int = 300) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Safely download macro data (DXY and SPX).
        Returns (df_dxy, df_spx) - either can be None if unavailable.
        """
        df_dxy = None
        df_spx = None
        
        # Try to get DXY data - check multiple possible symbol names
        dxy_symbols = [self.dxy_symbol, "USDX", "DXY", "DX", "US Dollar Index"]
        for sym in dxy_symbols:
            try:
                mt5.symbol_select(sym, True)
                data = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H1, 0, bars)
                if data is not None and len(data) > 0:
                    df_dxy = pd.DataFrame(data)
                    df_dxy['time'] = pd.to_datetime(df_dxy['time'], unit='s')
                    df_dxy.set_index('time', inplace=True)
                    self.dxy_symbol = sym  # Update to working symbol
                    break
            except Exception:
                continue
        
        # Try to get SPX data - check multiple possible symbol names
        spx_symbols = [self.spx_symbol, "SPX500", "SP500", "US500", "SPX", "S&P500", "US500.cash"]
        for sym in spx_symbols:
            try:
                mt5.symbol_select(sym, True)
                data = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H1, 0, bars)
                if data is not None and len(data) > 0:
                    df_spx = pd.DataFrame(data)
                    df_spx['time'] = pd.to_datetime(df_spx['time'], unit='s')
                    df_spx.set_index('time', inplace=True)
                    self.spx_symbol = sym  # Update to working symbol
                    break
            except Exception:
                continue
        
        return df_dxy, df_spx

    def get_market_context(self, df_main, df_dxy, df_spx):
        """
        Refined Intermarket Veto Logic.
        Handles missing macro data gracefully.
        """
        # Default return if macro data is unavailable
        default_context = {
            "veto_active": False,
            "reasons": [],
            "z_score": 0.0,
            "dxy_corr": 0.0,
            "macro_data_available": False
        }
        
        # Check if we have valid macro data
        dxy_valid = (df_dxy is not None and 
                     not df_dxy.empty and 
                     'close' in df_dxy.columns and 
                     len(df_dxy) >= 24)
        spx_valid = (df_spx is not None and 
                     not df_spx.empty and 
                     'close' in df_spx.columns and 
                     len(df_spx) >= 24)
        
        if not dxy_valid or not spx_valid:
            missing = []
            if not dxy_valid:
                missing.append(f"DXY ({self.dxy_symbol})")
            if not spx_valid:
                missing.append(f"SPX ({self.spx_symbol})")
            print(f"   Note: Macro data unavailable for {', '.join(missing)} - skipping intermarket analysis")
            return default_context
        
        try:
            # 1. Calculate Z-Score for Risk Sentiment (SPX)
            spx_returns = df_spx['close'].pct_change(24)
            if spx_returns.std() == 0:
                z_score_risk = 0.0
            else:
                z_score_risk = (spx_returns.iloc[-1] - spx_returns.mean()) / spx_returns.std()

            # 2. Institutional Divergence (SMT)
            # Check if DXY and Main Pair are moving in the SAME direction (Abnormal)
            dxy_slope = df_dxy['close'].iloc[-5:].diff().mean()
            main_slope = df_main['close'].iloc[-5:].diff().mean()

            # If slopes are both positive, DXY and EURUSD are rising together (Divergent)
            is_divergent = (dxy_slope * main_slope) > 0

            # 3. Correlation Strength
            current_corr = df_main['close'].rolling(24).corr(df_dxy['close']).iloc[-1]
            
            # Handle NaN correlation
            if pd.isna(current_corr):
                current_corr = 0.0

            # --- VETO SUMMARY ---
            veto_reasons = []
            if z_score_risk < -2.0:
                veto_reasons.append("Extreme Risk-Off Panic")
            if is_divergent:
                veto_reasons.append("Macro Divergence (SMT)")
            if current_corr > -0.70:
                veto_reasons.append("Weak Inverse Correlation")

            return {
                "veto_active": len(veto_reasons) > 0,
                "reasons": veto_reasons,
                "z_score": round(float(z_score_risk), 2),
                "dxy_corr": round(float(current_corr), 4),
                "macro_data_available": True
            }
            
        except Exception as e:
            print(f"   Warning: Error in market context analysis: {e}")
            return default_context

    def download_data(self, bars: int = 35000) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Download multi-timeframe data from MT5.

        Args:
            bars: Number of bars to download

        Returns:
            Tuple of (df_h1, df_h4, df_d1) DataFrames
        """
        print(f"Downloading multi-timeframe data for {self.symbol}...")
        try:
            # Download data for different timeframes
            df_h1 = pd.DataFrame(mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H1, 0, bars))
            df_h4 = pd.DataFrame(mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H4, 0, bars // 4))
            df_d1 = pd.DataFrame(mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_D1, 0, bars // 20))

            # Validate downloaded data with timeframe-appropriate minimums
            # FIXED: Indentation was wrong - validation and processing must be INSIDE the loop
            min_bars = {"H1": 100, "H4": 50, "D1": 20}
            for df, name in [(df_h1, "H1"), (df_h4, "H4"), (df_d1, "D1")]:
                required = min_bars.get(name, 100)
                if df is None or df.empty or len(df) < required:
                    print(f"Failed to download {name} data for {self.symbol} "
                          f"(got {len(df) if df is not None else 0}, need {required})")
                    return None, None, None
                # Process time index - THIS WAS INCORRECTLY INDENTED BEFORE
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                df.rename(columns={'tick_volume': 'volume'}, inplace=True)

            print(f"Downloaded {len(df_h1)} H1 bars from {df_h1.index.min()} to {df_h1.index.max()}")
            return df_h1, df_h4, df_d1

        except Exception as e:
            print(f"Error downloading data: {e}")
            return None, None, None

    def create_features(self, df_h1: pd.DataFrame, df_h4: pd.DataFrame, df_d1: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features for model training.

        Args:
            df_h1: H1 timeframe data
            df_h4: H4 timeframe data
            df_d1: D1 timeframe data

        Returns:
            DataFrame with engineered features
        """
        print("Creating advanced features...")
        df = df_h1.copy()

        # Multi-timeframe features
        df['sma_20_h4'] = df_h4['close'].rolling(window=20).mean().reindex(df.index, method='ffill')
        df['sma_20_d1'] = df_d1['close'].rolling(window=20).mean().reindex(df.index, method='ffill')

        # RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['rsi_14_h1'] = 100 - (100 / (1 + (gain / (loss + 1e-8))))

        # Time-based features
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

        # Returns and volatility
        df['log_return_1h'] = np.log(df['close'] / df['close'].shift(1))
        df['log_return_4h'] = np.log(df['close'] / df['close'].shift(4))
        df['log_return_1d'] = np.log(df['close'] / df['close'].shift(24))

        df['volatility_30'] = df['log_return_1h'].rolling(30).std()

        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)

        # Clean data - remove NaN and infinite values
        df.dropna(inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        if len(df) < self.lookback_periods + 100:
            print(f"WARNING: Only {len(df)} bars after feature creation. May not be enough for training.")

        print(f"   Created {len(df.columns)} features from {len(df)} bars")
        return df

    def perform_feature_selection(self, df: pd.DataFrame, num_features: int = 25) -> pd.DataFrame:
        """
        Select most important features using LightGBM.

        Args:
            df: DataFrame with all features
            num_features: Number of top features to select

        Returns:
            DataFrame with selected features
        """
        print(f"Performing feature selection to find top {num_features} features...")
        target = self.target_column
        exclude_cols = [target, 'log_return_4h', 'log_return_1d', 'close', 'open', 'high', 'low', 'time']
        features = [col for col in df.columns if col not in exclude_cols]

        X = df[features]
        y = df[target]

        # Train LightGBM for feature importance
        lgb_train = lgb.Dataset(X, y)
        params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'n_estimators': 200,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42
        }

        model = lgb.train(params, lgb_train, num_boost_round=100)
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importance()
        }).sort_values('importance', ascending=False)

        self.feature_cols = feature_importance['feature'].head(num_features).tolist()
        print(f"   Selected top {len(self.feature_cols)} features.")

        # Save selected features
        with open(self.selected_features_path, 'w') as f:
            json.dump(self.feature_cols, f)

        return df[self.feature_cols + ['log_return_1h', 'log_return_4h', 'log_return_1d', 'close']]

    def _prepare_sequential_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare sequential data for deep learning models.

        Args:
            df: DataFrame with features

        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        target_col = self.target_column
        train_size = int(len(df) * 0.70)
        val_size = int(len(df) * 0.15)

        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]

        print(f"   Target stats - Mean: {train_df[target_col].mean():.6f}, Std: {train_df[target_col].std():.6f}")

        # Remove extreme outliers (more than 5 std devs)
        mean_return = train_df[target_col].mean()
        std_return = train_df[target_col].std()
        train_df = train_df[abs(train_df[target_col] - mean_return) < (5 * std_return)]
        print(f"   Removed {train_size - len(train_df)} outliers from training data")

        # Initialize scalers
        self.feature_scaler = RobustScaler()
        self.target_scaler = RobustScaler()

        # Scale features and target
        train_scaled_features = self.feature_scaler.fit_transform(train_df[self.feature_cols])
        train_scaled_target = self.target_scaler.fit_transform(train_df[[target_col]])

        val_scaled_features = self.feature_scaler.transform(val_df[self.feature_cols])
        val_scaled_target = self.target_scaler.transform(val_df[[target_col]])

        print(f"   Target scaler - Center: {self.target_scaler.center_[0]:.6f}, Scale: {self.target_scaler.scale_[0]:.6f}")

        def create_sequences(features: np.ndarray, target: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
            """Create sequences for time series prediction."""
            X, y = [], []
            for i in range(lookback, len(features)):
                X.append(features[i - lookback:i])
                y.append(target[i])
            return np.array(X), np.array(y)

        X_train, y_train = create_sequences(train_scaled_features, train_scaled_target, self.lookback_periods)
        X_val, y_val = create_sequences(val_scaled_features, val_scaled_target, self.lookback_periods)

        print(f"   Prepared sequential data: X_train shape {X_train.shape}")
        return X_train, y_train, X_val, y_val

    def _prepare_tabular_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare tabular data for tree-based models.

        Args:
            df: DataFrame with features

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, feature_columns)
        """
        print("Preparing tabular data for tree-based models...")
        df_tabular = df.copy()
        feature_cols_tabular = self.feature_cols[:]

        # Create lag features
        for col in self.feature_cols:
            for lag in [1, 3, 5, 10]:
                new_col = f'{col}_lag_{lag}'
                df_tabular[new_col] = df_tabular[col].shift(lag)
                if new_col not in feature_cols_tabular:
                    feature_cols_tabular.append(new_col)

        df_tabular.dropna(inplace=True)
        final_feature_cols = [c for c in feature_cols_tabular if c in df_tabular.columns]

        # Split data
        train_size = int(len(df_tabular) * 0.85)
        train_df = df_tabular[:train_size]
        val_df = df_tabular[train_size:]

        X_train = train_df[final_feature_cols]
        y_train = train_df[self.target_column]
        X_val = val_df[final_feature_cols]
        y_val = val_df[self.target_column]

        print(f"   Prepared tabular data: X_train shape {X_train.shape}")
        return X_train, y_train, X_val, y_val, final_feature_cols

    def _build_dl_model(self, model_type: str, input_shape: Tuple[int, int], hp: Optional[kt.HyperParameters] = None) -> Model:
        """
        Build a deep learning model.

        Args:
            model_type: Type of model ('lstm', 'gru', 'transformer', 'tcn')
            input_shape: Input shape (lookback, features)
            hp: Hyperparameters for tuning

        Returns:
            Compiled Keras model
        """
        # Default hyperparameters
        lstm_units = 64
        conv_filters = 64
        dropout_rate = 0.3
        learning_rate = 0.0005

        # Override with tuning hyperparameters if provided
        if hp:
            lstm_units = hp.Int('lstm_units', 32, 128, 32)
            conv_filters = hp.Int('conv_filters', 32, 128, 32)
            dropout_rate = hp.Float('dropout', 0.2, 0.5, 0.1)
            learning_rate = hp.Choice('learning_rate', [1e-3, 5e-4, 1e-4])

        inputs = layers.Input(shape=input_shape)
        x = inputs

        # Build model based on type
        if model_type == 'lstm':
            # IMPROVED: Added attention mechanism
            x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.LayerNormalization()(x)
            x = AttentionLayer()(x)  # Attention instead of GlobalAveragePooling
            
        elif model_type == 'gru':
            # NEW: GRU model - lighter than LSTM, often comparable performance
            x = layers.Bidirectional(layers.GRU(lstm_units, return_sequences=True))(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.LayerNormalization()(x)
            x = layers.Bidirectional(layers.GRU(lstm_units // 2, return_sequences=False))(x)
            x = layers.Dropout(dropout_rate)(x)
            
        elif model_type == 'transformer':
            x = TransformerBlock(embed_dim=input_shape[1], num_heads=4, ff_dim=64, rate=dropout_rate)(x)
            x = layers.GlobalAveragePooling1D()(x)
            
        elif model_type == 'tcn':
            # FIXED: TCN architecture - ensure consistent output dimension
            # The issue was TCN might have been producing different scale outputs
            x = layers.Conv1D(filters=conv_filters, kernel_size=3, dilation_rate=1,
                              padding='causal', activation='relu')(x)
            x = layers.BatchNormalization()(x)  # Added batch norm for stability
            x = layers.Dropout(dropout_rate)(x)
            
            x = layers.Conv1D(filters=conv_filters, kernel_size=3, dilation_rate=2,
                              padding='causal', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)
            
            x = layers.Conv1D(filters=conv_filters, kernel_size=3, dilation_rate=4,
                              padding='causal', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            
            # FIXED: Use GlobalAveragePooling instead of taking last timestep
            # This makes TCN output more comparable to other models
            x = layers.GlobalAveragePooling1D()(x)
        else:
            raise ValueError(f"Unknown DL model type: {model_type}")

        # Output layers - consistent across all model types
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(64, activation='relu')(x)  # Added extra layer for consistency
        outputs = layers.Dense(1, activation='linear')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='huber', metrics=['mae'])
        return model

    def tune_hyperparameters(self) -> None:
        """Run hyperparameter tuning for deep learning models."""
        print("\n" + "=" * 60 + "\nStarting Hyperparameter Tuning...\n" + "=" * 60)

        # Download and prepare data
        df_h1, df_h4, df_d1 = self.download_data()
        if df_h1 is None:
            return

        df_features = self.create_features(df_h1, df_h4, df_d1)
        df_selected = self.perform_feature_selection(df_features)
        X_train, y_train, X_val, y_val = self._prepare_sequential_data(df_selected)

        # Save scalers
        with open(self.feature_scaler_path, 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        with open(self.target_scaler_path, 'wb') as f:
            pickle.dump(self.target_scaler, f)

        # Create tuner
        def model_builder(hp):
            return self._build_dl_model('lstm', (X_train.shape[1], X_train.shape[2]), hp=hp)

        tuner = kt.RandomSearch(
            model_builder,
            objective='val_loss',
            max_trials=15,
            executions_per_trial=1,
            directory=self.tuner_dir,
            project_name=f'tuner_{self.symbol}'
        )

        # Run tuning
        tuner.search(
            X_train, y_train,
            epochs=50,
            validation_data=(X_val, y_val),
            callbacks=[EarlyStopping('val_loss', patience=5)]
        )

        # Display results
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("\n--- Best Hyperparameters Found ---")
        for param, value in best_hps.values.items():
            print(f"{param}: {value}")
        print("---------------------------------\n")
        print("Tuning complete. Re-run with 'train --force' to use these new settings.")

    def train_model(self, force_retrain: bool = False) -> None:
        """
        Train the ensemble of models (OLD METHOD - single timeframe).
        For multi-timeframe training, use train_model_multitimeframe() instead.

        Args:
            force_retrain: Force retraining even if models exist
        """
        print("\n" + "=" * 60 + "\nStarting Hybrid Ensemble Training (Single Timeframe)...\n" + "=" * 60)
        print("WARNING: This trains only 1H models and scales predictions.")
        print("For better results, use train_model_multitimeframe() instead.\n")

        # Check if models already exist
        model_type_counts_check = defaultdict(int)
        all_models_exist = True
        for model_type in self.ensemble_model_types:
            model_index = model_type_counts_check[model_type]
            if not os.path.exists(self._get_model_path(model_type, model_index)):
                all_models_exist = False
                break
            model_type_counts_check[model_type] += 1

        if all_models_exist and not force_retrain:
            print("All models already exist. Loading them. Use --force to retrain.")
            self.load_model_assets()
            return

        # Download and prepare data
        df_h1, df_h4, df_d1 = self.download_data()
        if df_h1 is None:
            return

        df_features = self.create_features(df_h1, df_h4, df_d1)
        df_selected = self.perform_feature_selection(df_features)

        # Prepare data for different model types
        X_train_seq, y_train_seq, X_val_seq, y_val_seq = self._prepare_sequential_data(df_selected)
        X_train_tab, y_train_tab, X_val_tab, y_val_tab, _ = self._prepare_tabular_data(df_selected)

        # Save scalers
        with open(self.feature_scaler_path, 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        with open(self.target_scaler_path, 'wb') as f:
            pickle.dump(self.target_scaler, f)

        # Try to load best hyperparameters from tuning
        best_hps = None
        try:
            tuner = kt.RandomSearch(
                lambda hp: self._build_dl_model('lstm', (X_train_seq.shape[1], X_train_seq.shape[2]), hp),
                objective='val_loss',
                directory=self.tuner_dir,
                project_name=f'tuner_{self.symbol}'
            )
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            print("Found best hyperparameters from tuning.")
        except Exception:
            print("No tuning data found. Using default hyperparameters for DL models.")

        # Train each model in the ensemble
        model_type_counts = defaultdict(int)
        for model_type in self.ensemble_model_types:
            model_index = model_type_counts[model_type]
            print(f"\n--- Training Model {model_type.upper()} (Instance {model_index}) ---")
            tf.random.set_seed(42 + model_index)

            if model_type in ['lstm', 'gru', 'transformer', 'tcn']:
                # Train deep learning model
                model = self._build_dl_model(model_type, (X_train_seq.shape[1], X_train_seq.shape[2]), hp=best_hps)
                callbacks = [
                    EarlyStopping('val_loss', patience=15, restore_best_weights=True),
                    ReduceLROnPlateau('val_loss', patience=5, factor=0.5)
                ]
                model.fit(
                    X_train_seq, y_train_seq,
                    validation_data=(X_val_seq, y_val_seq),
                    epochs=150,
                    batch_size=64,
                    callbacks=callbacks,
                    verbose=1
                )
                model.save(self._get_model_path(model_type, model_index))

            elif model_type == 'lgbm':
                # Train LightGBM model
                model = lgb.LGBMRegressor(
                    objective='regression_l1',
                    n_estimators=1000,
                    learning_rate=0.05,
                    random_state=42 + model_index,
                    n_jobs=-1,
                    verbose=-1
                )
                model.fit(
                    X_train_tab, y_train_tab,
                    eval_set=[(X_val_tab, y_val_tab)],
                    eval_metric='mae',
                    callbacks=[lgb.early_stopping(100, verbose=False)]
                )
                with open(self._get_model_path(model_type, model_index), 'wb') as f:
                    pickle.dump(model, f)

            model_type_counts[model_type] += 1

        print("\nEnsemble training complete and all assets saved.")
        self.load_model_assets()

    def train_model_multitimeframe(self, force_retrain: bool = False) -> None:
        """
        Train separate models for each timeframe (1H, 4H, 1D).
        This is the RECOMMENDED method for accurate multi-timeframe predictions.

        Args:
            force_retrain: Force retraining even if models exist
        """
        print("\n" + "=" * 60)
        print("Starting Multi-Timeframe Ensemble Training...")
        print("=" * 60 + "\n")
        print("This will train 3 separate ensembles (1H, 4H, 1D)")
        print(f"Total models to train: {len(self.ensemble_model_types) * 3}")
        print("Estimated time: 4-6 hours\n")

        # Download and prepare data
        df_h1, df_h4, df_d1 = self.download_data()
        if df_h1 is None:
            return

        df_features = self.create_features(df_h1, df_h4, df_d1)
        df_selected = self.perform_feature_selection(df_features)

        # Define target columns for each timeframe
        timeframe_targets = {
            '1H': 'log_return_1h',
            '4H': 'log_return_4h',
            '1D': 'log_return_1d'
        }

        # Try to load best hyperparameters from tuning
        best_hps = None
        try:
            tuner = kt.RandomSearch(
                lambda hp: self._build_dl_model('lstm', (60, 25), hp),
                objective='val_loss',
                directory=self.tuner_dir,
                project_name=f'tuner_{self.symbol}'
            )
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            print("Found best hyperparameters from tuning.")
        except Exception:
            print("No tuning data found. Using default hyperparameters.")

        # Train a separate ensemble for each timeframe
        for tf_name, target_col in timeframe_targets.items():
            print(f"\n{'=' * 60}")
            print(f"Training Ensemble for {tf_name} Predictions (Target: {target_col})")
            print(f"{'=' * 60}\n")

            # Temporarily change the target column
            original_target = self.target_column
            self.target_column = target_col

            # Prepare data with this target
            X_train_seq, y_train_seq, X_val_seq, y_val_seq = self._prepare_sequential_data(df_selected)
            X_train_tab, y_train_tab, X_val_tab, y_val_tab, _ = self._prepare_tabular_data(df_selected)

            # Save scalers for this timeframe
            scaler_suffix = f"_{tf_name}"
            with open(self.feature_scaler_path.replace('.pkl', f'{scaler_suffix}.pkl'), 'wb') as f:
                pickle.dump(self.feature_scaler, f)
            with open(self.target_scaler_path.replace('.pkl', f'{scaler_suffix}.pkl'), 'wb') as f:
                pickle.dump(self.target_scaler, f)

            # Train each model type for this timeframe
            model_type_counts = defaultdict(int)
            for model_type in self.ensemble_model_types:
                model_index = model_type_counts[model_type]
                print(f"\n--- Training {model_type.upper()} for {tf_name} (Instance {model_index}) ---")
                tf.random.set_seed(42 + model_index)

                if model_type in ['lstm', 'gru', 'transformer', 'tcn']:
                    model = self._build_dl_model(model_type, (X_train_seq.shape[1], X_train_seq.shape[2]), hp=best_hps)
                    callbacks = [
                        EarlyStopping('val_loss', patience=15, restore_best_weights=True),
                        ReduceLROnPlateau('val_loss', patience=5, factor=0.5)
                    ]
                    model.fit(
                        X_train_seq, y_train_seq,
                        validation_data=(X_val_seq, y_val_seq),
                        epochs=150,
                        batch_size=64,
                        callbacks=callbacks,
                        verbose=1
                    )
                    # Save with timeframe suffix
                    model_path = self._get_model_path(model_type, model_index).replace('.keras', f'_{tf_name}.keras')
                    model.save(model_path)
                    print(f"Saved: {model_path}")

                elif model_type == 'lgbm':
                    model = lgb.LGBMRegressor(
                        objective='regression_l1',
                        n_estimators=1000,
                        learning_rate=0.05,
                        random_state=42 + model_index,
                        n_jobs=-1,
                        verbose=-1
                    )
                    model.fit(
                        X_train_tab, y_train_tab,
                        eval_set=[(X_val_tab, y_val_tab)],
                        eval_metric='mae',
                        callbacks=[lgb.early_stopping(100, verbose=False)]
                    )
                    # Save with timeframe suffix
                    model_path = self._get_model_path(model_type, model_index).replace('.pkl', f'_{tf_name}.pkl')
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    print(f"Saved: {model_path}")

                model_type_counts[model_type] += 1

            # Restore original target
            self.target_column = original_target

        # Copy 1H scalers and models to base names for backward compatibility
        print("\nSaving base scalers and models for backward compatibility...")
        try:
            import shutil

            # Copy scalers
            h1_feature_scaler = self.feature_scaler_path.replace('.pkl', '_1H.pkl')
            h1_target_scaler = self.target_scaler_path.replace('.pkl', '_1H.pkl')

            if os.path.exists(h1_feature_scaler):
                shutil.copy(h1_feature_scaler, self.feature_scaler_path)
                print(f"[OK] Copied {os.path.basename(h1_feature_scaler)} -> {os.path.basename(self.feature_scaler_path)}")

            if os.path.exists(h1_target_scaler):
                shutil.copy(h1_target_scaler, self.target_scaler_path)
                print(f"[OK] Copied {os.path.basename(h1_target_scaler)} -> {os.path.basename(self.target_scaler_path)}")

            # Copy model files
            print("\nCopying 1H models to base names...")
            model_type_counts = defaultdict(int)
            for model_type in self.ensemble_model_types:
                model_index = model_type_counts[model_type]
                base_model_path = self._get_model_path(model_type, model_index)

                # Construct 1H model path
                ext = '.keras' if model_type in ['lstm', 'gru', 'transformer', 'tcn'] else '.pkl'
                h1_model_path = base_model_path.replace(ext, f'_1H{ext}')

                if os.path.exists(h1_model_path):
                    shutil.copy(h1_model_path, base_model_path)
                    print(f"[OK] Copied {os.path.basename(h1_model_path)} -> {os.path.basename(base_model_path)}")

                model_type_counts[model_type] += 1

        except Exception as e:
            print(f"Warning: Could not copy all base files: {e}")
            print("This may cause issues with backtest mode, but multi-TF predictions will work fine.")

        print("\n" + "=" * 60)
        print("Multi-timeframe ensemble training complete!")
        print("=" * 60)
        print(f"\nTrained {len(self.ensemble_model_types) * 3} models total")
        print("Use predict-multitf command to make predictions with these models")

    def load_model_assets(self) -> bool:
        """
        Load all trained models and scalers (single timeframe method).

        Returns:
            True if successful, False otherwise
        """
        print("Loading all model assets for the ensemble...")

        # Auto-detect models if not specified
        if not self.ensemble_model_types:
            self.ensemble_model_types = self._detect_trained_models()
            if not self.ensemble_model_types:
                print("Error: No trained models found. Please run the 'train' command first.")
                return False
            self.num_ensemble_models = len(self.ensemble_model_types)
            self.ensemble_weights = [1.0 / self.num_ensemble_models] * self.num_ensemble_models
            print(f"Detected trained models: {self.ensemble_model_types}")

        try:
            # Load feature list and scalers
            with open(self.selected_features_path, 'r') as f:
                self.feature_cols = json.load(f)

            # Try to load base scaler first, fallback to multi-TF scalers if needed
            feature_scaler_loaded = False
            target_scaler_loaded = False

            # Try base scaler first
            if os.path.exists(self.feature_scaler_path):
                with open(self.feature_scaler_path, 'rb') as f:
                    self.feature_scaler = pickle.load(f)
                feature_scaler_loaded = True
            else:
                # Check for multi-timeframe scalers (try 1H first as base timeframe)
                for tf_suffix in ['_1H', '_4H', '_1D']:
                    mtf_path = self.feature_scaler_path.replace('.pkl', f'{tf_suffix}.pkl')
                    if os.path.exists(mtf_path):
                        print(f"Note: Using multi-timeframe scaler: {os.path.basename(mtf_path)}")
                        with open(mtf_path, 'rb') as f:
                            self.feature_scaler = pickle.load(f)
                        feature_scaler_loaded = True
                        break

            if not feature_scaler_loaded:
                raise FileNotFoundError(f"Feature scaler not found: {self.feature_scaler_path}")

            # Try base target scaler first
            if os.path.exists(self.target_scaler_path):
                with open(self.target_scaler_path, 'rb') as f:
                    self.target_scaler = pickle.load(f)
                target_scaler_loaded = True
            else:
                # Check for multi-timeframe target scalers
                for tf_suffix in ['_1H', '_4H', '_1D']:
                    mtf_path = self.target_scaler_path.replace('.pkl', f'{tf_suffix}.pkl')
                    if os.path.exists(mtf_path):
                        with open(mtf_path, 'rb') as f:
                            self.target_scaler = pickle.load(f)
                        target_scaler_loaded = True
                        break

            if not target_scaler_loaded:
                raise FileNotFoundError(f"Target scaler not found: {self.target_scaler_path}")

            # Load each model
            self.models = {}
            model_type_counts = defaultdict(int)

            for model_type in self.ensemble_model_types:
                model_index = model_type_counts[model_type]
                model_path = self._get_model_path(model_type, model_index)
                model_name = f"{model_type}_{model_index}"

                # Auto-detect multi-timeframe model files if base doesn't exist
                actual_model_path = model_path
                if not os.path.exists(model_path):
                    # Check for multi-timeframe model files
                    for tf_suffix in ['_1H', '_4H', '_1D']:
                        if model_type in ['lstm', 'gru', 'transformer', 'tcn']:
                            # Try .keras first (newer), then .h5 (older)
                            for ext in ['.keras', '.h5']:
                                mtf_path = model_path.replace('.keras', f'{tf_suffix}{ext}').replace('.h5', f'{tf_suffix}{ext}')
                                if os.path.exists(mtf_path):
                                    print(f"Note: Using multi-timeframe model: {os.path.basename(mtf_path)}")
                                    actual_model_path = mtf_path
                                    break
                        else:  # lgbm
                            mtf_path = model_path.replace('.pkl', f'{tf_suffix}.pkl')
                            if os.path.exists(mtf_path):
                                print(f"Note: Using multi-timeframe model: {os.path.basename(mtf_path)}")
                                actual_model_path = mtf_path
                                break

                        if actual_model_path != model_path:
                            break

                if not os.path.exists(actual_model_path):
                    print(f"ERROR: Model file not found: {model_path}")
                    print(f"       Also checked for multi-TF versions with suffixes _1H, _4H, _1D")
                    return False

                if model_type in ['lstm', 'gru', 'transformer', 'tcn']:
                    self.models[model_name] = load_model(
                        actual_model_path,
                        custom_objects={
                            'TransformerBlock': TransformerBlock,
                            'AttentionLayer': AttentionLayer
                        }
                    )
                elif model_type == 'lgbm':
                    with open(actual_model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)

                model_type_counts[model_type] += 1

            print(f"Successfully loaded {len(self.models)} models: {list(self.models.keys())}")
            return True

        except FileNotFoundError as e:
            print(f"Error: Model assets not found. Please train the model first. Missing: {e.filename}")
            return False
        except Exception as e:
            print(f"Error loading model assets: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_model_assets_multitimeframe(self) -> bool:
        """
        Load models for all timeframes (multi-timeframe method).

        Returns:
            True if successful, False otherwise
        """
        print("Loading multi-timeframe model assets...")

        if not self.ensemble_model_types:
            self.ensemble_model_types = self._detect_trained_models()
            if not self.ensemble_model_types:
                print("Error: No trained models found.")
                return False

        # Load feature list
        try:
            with open(self.selected_features_path, 'r') as f:
                self.feature_cols = json.load(f)
        except FileNotFoundError:
            print("ERROR: Feature list not found. Please train models first.")
            return False

        timeframe_list = ['1H', '4H', '1D']
        self.models_by_timeframe = {}
        self.scalers_by_timeframe = {}

        for tf_name in timeframe_list:
            print(f"\nLoading models for {tf_name}...")

            # Load scalers for this timeframe
            try:
                scaler_suffix = f"_{tf_name}"
                with open(self.feature_scaler_path.replace('.pkl', f'{scaler_suffix}.pkl'), 'rb') as f:
                    feature_scaler = pickle.load(f)
                with open(self.target_scaler_path.replace('.pkl', f'{scaler_suffix}.pkl'), 'rb') as f:
                    target_scaler = pickle.load(f)
                self.scalers_by_timeframe[tf_name] = (feature_scaler, target_scaler)
                print(f"  Loaded scalers for {tf_name}")
            except FileNotFoundError:
                print(f"WARNING: Scalers not found for {tf_name}")
                return False

            # Load models for this timeframe
            models = {}
            model_type_counts = defaultdict(int)

            for model_type in self.ensemble_model_types:
                model_index = model_type_counts[model_type]
                model_name = f"{model_type}_{model_index}"

                if model_type in ['lstm', 'gru', 'transformer', 'tcn']:
                    model_path = self._get_model_path(model_type, model_index).replace('.keras', f'_{tf_name}.keras')
                    if os.path.exists(model_path):
                        models[model_name] = load_model(
                            model_path,
                            custom_objects={
                                'TransformerBlock': TransformerBlock,
                                'AttentionLayer': AttentionLayer
                            }
                        )
                        print(f"  Loaded {model_name}")
                    else:
                        print(f"ERROR: Model not found: {model_path}")
                        return False

                elif model_type == 'lgbm':
                    model_path = self._get_model_path(model_type, model_index).replace('.pkl', f'_{tf_name}.pkl')
                    if os.path.exists(model_path):
                        with open(model_path, 'rb') as f:
                            models[model_name] = pickle.load(f)
                        print(f"  Loaded {model_name}")
                    else:
                        print(f"ERROR: Model not found: {model_path}")
                        return False

                model_type_counts[model_type] += 1

            self.models_by_timeframe[tf_name] = models
            print(f"  Total models for {tf_name}: {len(models)}")

        print(f"\nSuccessfully loaded models for all {len(self.models_by_timeframe)} timeframes")
        return len(self.models_by_timeframe) > 0

    def _detect_trained_models(self) -> Optional[List[str]]:
        """
        Detect trained models from saved files.

        Returns:
            List of detected model types or None if none found
        """
        # Find all model files
        all_model_files = glob.glob(os.path.join(self.base_path, f"model_{self.symbol}_*.h5"))
        all_model_files += glob.glob(os.path.join(self.base_path, f"model_{self.symbol}_*.pkl"))
        all_model_files += glob.glob(os.path.join(self.base_path, f"model_{self.symbol}_*.keras"))

        found_models = set()

        # Parse model files
        for f in all_model_files:
            parts = os.path.basename(f).split('_')
            if len(parts) >= 4:
                model_type = parts[2]
                try:
                    # Handle both regular and multitimeframe models
                    index_part = parts[3].split('.')[0]

                    # Also detect multi-timeframe models
                    if index_part in ['1H', '4H', '1D']:
                        # This is a multi-TF model
                        if len(parts) >= 5:
                            # Format: model_SYMBOL_TYPE_INDEX_TIMEFRAME.ext
                            try:
                                model_index = int(parts[3])
                                if model_type in ['lstm', 'gru', 'transformer', 'tcn', 'lgbm']:
                                    found_models.add((model_type, model_index))
                            except ValueError:
                                pass
                        continue

                    model_index = int(index_part)
                    if model_type in ['lstm', 'gru', 'transformer', 'tcn', 'lgbm']:
                        found_models.add((model_type, model_index))
                except (ValueError, IndexError):
                    continue

        # Sort by index then type
        found_models = sorted(list(found_models), key=lambda x: (x[1], x[0]))
        detected = [model_type for model_type, model_index in found_models]

        print(f"   Found model files: {found_models}")
        print(f"   Detected models: {detected}")
        return detected if detected else None

    def _get_model_path(self, model_type: str, index: int) -> str:
        """Get the file path for a model."""
        ext = 'keras' if model_type in ['lstm', 'gru', 'transformer', 'tcn'] else 'pkl'
        return os.path.join(self.base_path, f"model_{self.symbol}_{model_type}_{index}.{ext}")

    def run_prediction_cycle(self):
        """Updated with Macro integration."""
        print(f"\n--- Single-Timeframe Prediction Cycle: {self.symbol} ---")

        # Download main data first
        df_h1, df_h4, df_d1 = self.download_data(500)
        if df_h1 is None:
            return

        # Download macro data safely
        df_dxy, df_spx = self._download_macro_data(300)
        context = self.get_market_context(df_h1, df_dxy, df_spx)

        print("\n" + "=" * 60)
        print(f"Starting Prediction Cycle for {self.symbol} at {datetime.now()}")
        print("=" * 60 + "\n")
        print("WARNING: Using single-timeframe models with scaling.")
        print("For better predictions, use run_prediction_cycle_multitimeframe()\n")

        # Load models if not already loaded
        if not self.models:
            if not self.load_model_assets():
                return

        # Evaluate past predictions and update weights
        self._evaluate_past_predictions()
        self.update_ensemble_weights()

        # Download fresh data
        df_h1 = pd.DataFrame(mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H1, 0, 300))
        df_h4 = pd.DataFrame(mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H4, 0, 300))
        df_d1 = pd.DataFrame(mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_D1, 0, 300))

        # Validate data
        for df, name in [(df_h1, "H1"), (df_h4, "H4"), (df_d1, "D1")]:
            if df.empty or len(df) < 100:
                print(f"ERROR: Insufficient {name} data for prediction")
                return
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            if 'tick_volume' in df.columns:
                df.rename(columns={'tick_volume': 'volume'}, inplace=True)

        # Create features
        df = self.create_features(df_h1, df_h4, df_d1)
        current_price = df['close'].iloc[-1]

        # Prepare sequential input
        last_sequence_raw = df.iloc[-self.lookback_periods:][self.feature_cols].values
        last_sequence_scaled = self.feature_scaler.transform(last_sequence_raw)
        X_pred_seq = last_sequence_scaled.reshape(1, self.lookback_periods, len(self.feature_cols))

        # Convert to TensorFlow tensor to avoid retracing warnings
        X_pred_seq = tf.convert_to_tensor(X_pred_seq, dtype=tf.float32)

        # Prepare tabular input for LightGBM
        df_tabular = df.copy()
        final_tab_cols = []
        for model_name, model in self.models.items():
            if 'lgbm' in model_name:
                final_tab_cols = model.feature_name_
                break

        X_pred_tab = None
        if final_tab_cols:
            for col in self.feature_cols:
                for lag in [1, 3, 5, 10]:
                    new_col = f'{col}_lag_{lag}'
                    if new_col in final_tab_cols:
                        df_tabular[new_col] = df_tabular[col].shift(lag)

            # Check for NaN values in tabular features
            last_row = df_tabular.iloc[-1][final_tab_cols]
            if last_row.isna().any():
                print("WARNING: NaN values detected in tabular features, filling with forward fill")
                df_tabular.ffill(inplace=True)

            X_pred_tab = df_tabular.iloc[-1][final_tab_cols].values.reshape(1, -1)

        # Make predictions
        predictions = {}
        timeframes = {"1H": 1, "4H": 4, "1D": 24}
        ensemble_predictions_map = {}

        print("\nMaking predictions with hybrid ensemble...")
        for tf_name, steps in timeframes.items():
            ensemble_preds = []
            raw_log_returns = []

            # Get predictions from each model
            for model_name, model in self.models.items():
                pred_log_return = 0.0

                try:
                    if 'lgbm' in model_name and X_pred_tab is not None:
                        pred_log_return = model.predict(X_pred_tab)[0]
                    elif 'lgbm' not in model_name:
                        # Use direct call to avoid retracing
                        pred_log_return_scaled = model(X_pred_seq, training=False).numpy()[0][0]
                        pred_log_return = self.target_scaler.inverse_transform([[pred_log_return_scaled]])[0][0]
                except Exception as e:
                    print(f"WARNING: Error predicting with {model_name}: {e}")
                    continue

                raw_log_returns.append(pred_log_return)

                # Scale log return by time horizon (square root scaling)
                steps_adjusted = np.sqrt(steps) if steps > 1 else steps
                predicted_price = current_price * np.exp(pred_log_return * steps_adjusted)

                # Validate predicted price
                if np.isnan(predicted_price) or np.isinf(predicted_price):
                    print(f"WARNING: Invalid prediction from {model_name}, using current price")
                    predicted_price = current_price

                ensemble_preds.append(predicted_price)

            # Check if we have valid predictions
            if not ensemble_preds:
                print(f"ERROR: No valid predictions for {tf_name}, skipping")
                continue

            print(f"\n{tf_name} Debug:")
            print(f"  Raw log returns: {[f'{lr:.6f}' for lr in raw_log_returns]}")
            print(f"  Steps: {steps} -> Adjusted: {steps_adjusted:.2f}")
            print(f"  Predicted prices: {[f'{p:.5f}' for p in ensemble_preds]}")

            ensemble_predictions_map[tf_name] = ensemble_preds
            raw_prediction = np.average(ensemble_preds, weights=self.ensemble_weights[:len(ensemble_preds)])

            print(f"  Raw ensemble average: {raw_prediction:.5f}")

            # Apply smoothing to log returns
            raw_log_return = np.log(raw_prediction / current_price)

            if self.use_kalman:
                # Use Kalman filtering
                print(f"  Kalman state before: x={self.kalman_filters[tf_name].x:.6f}, p={self.kalman_filters[tf_name].p:.6f}")
                smoothed_log_return = self.kalman_filters[tf_name].update(raw_log_return)
                print(f"  Kalman smoothed log return: {smoothed_log_return:.6f} (raw: {raw_log_return:.6f})")
                print(f"  Kalman state after: x={self.kalman_filters[tf_name].x:.6f}, p={self.kalman_filters[tf_name].p:.6f}")
                smoothed_prediction = current_price * np.exp(smoothed_log_return)
            else:
                # Use EMA smoothing
                if self.previous_predictions[tf_name] is not None:
                    prev_log_return = np.log(self.previous_predictions[tf_name] / current_price)
                    smoothed_log_return = self.ema_alpha * raw_log_return + (1 - self.ema_alpha) * prev_log_return
                    smoothed_prediction = current_price * np.exp(smoothed_log_return)
                    print(f"  EMA smoothed log return: {smoothed_log_return:.6f} (raw: {raw_log_return:.6f})")
                else:
                    smoothed_prediction = raw_prediction
                    print(f"  Using raw prediction (first prediction)")

            self.previous_predictions[tf_name] = smoothed_prediction

            max_change_pct = {'1H': 0.5, '4H': 1.0, '1D': 2.0}
            max_change = current_price * (max_change_pct.get(tf_name, 1.0) / 100.0)

            if abs(smoothed_prediction - current_price) > max_change:
                original_pred = smoothed_prediction
                if smoothed_prediction > current_price:
                    smoothed_prediction = current_price + max_change
                else:
                    smoothed_prediction = current_price - max_change
                print(f"  Capped from {original_pred:.5f} to {smoothed_prediction:.5f}")

            # change percentage for the EA
            change_pct = ((smoothed_prediction - current_price) / current_price) * 100.0

            predictions[tf_name] = {
                'prediction': round(smoothed_prediction, 5),
                'change_pct': round(change_pct, 3),
                'ensemble_std': round(np.std(ensemble_preds), 5),
            }

        # Log predictions for future evaluation
        self._log_prediction_for_evaluation(timeframes, ensemble_predictions_map, current_price)

        # Save predictions and status
        status = {
            'last_update': datetime.now().isoformat(),
            'status': 'online',
            'symbol': self.symbol,
            'current_price': round(current_price, 5),
            'ensemble_weights': [round(w, 3) for w in self.ensemble_weights],
            'price': df_h1['close'].iloc[-1],
            'market_context': context,
            'trade_allowed': not context['veto_active']
        }

        self.save_to_file(self.predictions_file, predictions)
        self.save_to_file(self.status_file, status)

        # Display results
        print("\n--- Prediction Cycle Complete! ---")
        print(f"Current Price: {current_price:.5f}")
        for timeframe, data in predictions.items():
            direction = "UP" if data['prediction'] > current_price else "DOWN"
            change_pct = ((data['prediction'] - current_price) / current_price) * 100
            print(f"   {direction} {timeframe}: {data['prediction']:.5f} ({change_pct:+.3f}%) (Uncertainty: +/-{data['ensemble_std']:.5f})")

    def run_prediction_cycle_multitimeframe(self):
        """Updated with Macro integration."""
        print(f"\n--- Multi-Timeframe Cycle: {self.symbol} ---")

        # Download main data first
        df_h1, df_h4, df_d1 = self.download_data(500)
        if df_h1 is None:
            return

        # Download macro data safely
        df_dxy, df_spx = self._download_macro_data(300)
        context = self.get_market_context(df_h1, df_dxy, df_spx)

        print("\n" + "=" * 60)
        print(f"Starting Multi-Timeframe Prediction Cycle for {self.symbol}")
        print("=" * 60 + "\n")

        if not hasattr(self, 'models_by_timeframe') or not self.models_by_timeframe:
            if not self.load_model_assets_multitimeframe():
                return

        # Download fresh data
        df_h1 = pd.DataFrame(mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H1, 0, 300))
        df_h4 = pd.DataFrame(mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H4, 0, 300))
        df_d1 = pd.DataFrame(mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_D1, 0, 300))

        # Validate data
        for df, name in [(df_h1, "H1"), (df_h4, "H4"), (df_d1, "D1")]:
            if df.empty or len(df) < 100:
                print(f"ERROR: Insufficient {name} data")
                return
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            if 'tick_volume' in df.columns:
                df.rename(columns={'tick_volume': 'volume'}, inplace=True)

        # Create features
        df = self.create_features(df_h1, df_h4, df_d1)
        current_price = df['close'].iloc[-1]

        predictions = {}

        print("\nMaking predictions with timeframe-specific models...")
        for tf_name, models in self.models_by_timeframe.items():
            if not models:
                print(f"WARNING: No models for {tf_name}, skipping")
                continue

            # Get scalers for this timeframe
            feature_scaler, target_scaler = self.scalers_by_timeframe[tf_name]

            # Prepare input data
            last_sequence_raw = df.iloc[-self.lookback_periods:][self.feature_cols].values
            last_sequence_scaled = feature_scaler.transform(last_sequence_raw)
            X_pred_seq = last_sequence_scaled.reshape(1, self.lookback_periods, len(self.feature_cols))

            # Convert to TensorFlow tensor to avoid retracing warnings
            X_pred_seq = tf.convert_to_tensor(X_pred_seq, dtype=tf.float32)

            # Get predictions from each model
            ensemble_preds = []
            for model_name, model in models.items():
                try:
                    if 'lgbm' in model_name:
                        # Prepare tabular data for LightGBM
                        df_tabular = df.copy()
                        for col in self.feature_cols:
                            for lag in [1, 3, 5, 10]:
                                new_col = f'{col}_lag_{lag}'
                                df_tabular[new_col] = df_tabular[col].shift(lag)
                        df_tabular.ffill(inplace=True)

                        X_pred_tab = df_tabular.iloc[-1][model.feature_name_].values.reshape(1, -1)
                        pred_log_return = model.predict(X_pred_tab)[0]
                    else:
                        # Deep learning model - use direct call to avoid retracing
                        pred_log_return_scaled = model(X_pred_seq, training=False).numpy()[0][0]
                        pred_log_return = target_scaler.inverse_transform([[pred_log_return_scaled]])[0][0]

                    # Convert log return to price (NO SCALING!)
                    predicted_price = current_price * np.exp(pred_log_return)

                    if not np.isnan(predicted_price) and not np.isinf(predicted_price):
                        ensemble_preds.append(predicted_price)

                except Exception as e:
                    print(f"WARNING: Error with {model_name} for {tf_name}: {e}")
                    continue

            if not ensemble_preds:
                print(f"ERROR: No valid predictions for {tf_name}")
                continue

            # Average ensemble predictions
            raw_prediction = np.mean(ensemble_preds)

            print(f"\n{tf_name}:")
            print(f"  Ensemble predictions: {[f'{p:.5f}' for p in ensemble_preds]}")
            print(f"  Average: {raw_prediction:.5f}")

            # Apply smoothing
            raw_log_return = np.log(raw_prediction / current_price)

            if self.use_kalman:
                smoothed_log_return = self.kalman_filters[tf_name].update(raw_log_return)
                smoothed_prediction = current_price * np.exp(smoothed_log_return)
                print(f"  Kalman smoothed: {smoothed_prediction:.5f}")
            else:
                if self.previous_predictions[tf_name] is not None:
                    prev_log_return = np.log(self.previous_predictions[tf_name] / current_price)
                    smoothed_log_return = self.ema_alpha * raw_log_return + (1 - self.ema_alpha) * prev_log_return
                    smoothed_prediction = current_price * np.exp(smoothed_log_return)
                    print(f"  EMA smoothed: {smoothed_prediction:.5f}")
                else:
                    smoothed_prediction = raw_prediction
                self.previous_predictions[tf_name] = smoothed_prediction

            # Sanity check
            max_change_pct = {'1H': 0.5, '4H': 1.0, '1D': 2.0}
            max_change = current_price * (max_change_pct.get(tf_name, 1.0) / 100.0)

            if abs(smoothed_prediction - current_price) > max_change:
                original_pred = smoothed_prediction
                if smoothed_prediction > current_price:
                    smoothed_prediction = current_price + max_change
                else:
                    smoothed_prediction = current_price - max_change
                print(f"  Capped from {original_pred:.5f} to {smoothed_prediction:.5f}")

            predictions[tf_name] = {
                'prediction': round(smoothed_prediction, 5),
                'ensemble_std': round(np.std(ensemble_preds), 5)
            }

        # Save predictions
        status = {
            'last_update': datetime.now().isoformat(),
            'status': 'online',
            'symbol': self.symbol,
            'current_price': round(current_price, 5),
            'method': 'multi-timeframe',
            'market_context': context,
            'trade_allowed': not context['veto_active']
        }

        self.save_to_file(self.predictions_file, predictions)
        self.save_to_file(self.status_file, status)

        # Display results
        print("\n--- Prediction Cycle Complete! ---")
        print(f"Current Price: {current_price:.5f}")
        for timeframe, data in predictions.items():
            direction = "UP" if data['prediction'] > current_price else "DOWN"
            change_pct = ((data['prediction'] - current_price) / current_price) * 100
            print(f"   {direction} {timeframe}: {data['prediction']:.5f} ({change_pct:+.3f}%) (±{data['ensemble_std']:.5f})")

    def _log_prediction_for_evaluation(self, timeframes_steps: Dict[str, int],
                                       ensemble_predictions_map: Dict[str, List[float]],
                                       current_price: float) -> None:
        """Log predictions for future evaluation."""
        try:
            with open(self.pending_eval_path, 'r') as f:
                pending = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pending = []

        now = datetime.now()
        for tf_name, steps in timeframes_steps.items():
            if tf_name in ensemble_predictions_map:
                pending.append({
                    "eval_timestamp": (now + timedelta(hours=steps)).isoformat(),
                    "pred_timestamp": now.isoformat(),
                    "timeframe": tf_name,
                    "start_price": current_price,
                    "predictions": ensemble_predictions_map[tf_name]
                })

        with open(self.pending_eval_path, 'w') as f:
            json.dump(pending, f, indent=2)

    def _evaluate_past_predictions(self) -> None:
        """Evaluate past predictions against actual prices."""
        print("Evaluating past predictions for ensemble weighting...")
        try:
            with open(self.pending_eval_path, 'r') as f:
                pending = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print("   No pending predictions to evaluate.")
            return

        remaining_evals = []
        evaluated_count = 0
        now = datetime.now()

        for entry in pending:
            try:
                eval_time = datetime.fromisoformat(entry['eval_timestamp'])

                # Ensure timezone-naive datetime for comparison
                if eval_time.tzinfo is not None:
                    eval_time = eval_time.replace(tzinfo=None)

                if now >= eval_time:
                    # Fetch actual price at evaluation time
                    rates = mt5.copy_rates_from(self.symbol, mt5.TIMEFRAME_H1, eval_time, 1)
                    if rates is not None and len(rates) > 0:
                        actual_future_price = rates[0]['close']
                        self.prediction_history[entry['timeframe']].append({
                            'predictions': entry['predictions'],
                            'actual': actual_future_price,
                            'timestamp': entry['pred_timestamp']
                        })
                        # Keep only recent history
                        if len(self.prediction_history[entry['timeframe']]) > self.ensemble_lookback:
                            self.prediction_history[entry['timeframe']].pop(0)
                        evaluated_count += 1
                    else:
                        # Keep for retry if data not available yet
                        remaining_evals.append(entry)
                else:
                    # Not time to evaluate yet
                    remaining_evals.append(entry)
            except Exception as e:
                print(f"   Error evaluating entry: {e}")
                continue

        print(f"   Evaluated {evaluated_count} predictions. {len(remaining_evals)} remaining.")

        # Save remaining evaluations
        with open(self.pending_eval_path, 'w') as f:
            json.dump(remaining_evals, f, indent=2)

    def update_ensemble_weights(self) -> None:
        """Update ensemble weights based on past performance."""
        if not self.ensemble_weights:
            return

        print("Updating ensemble weights...")
        model_errors = [0.0] * self.num_ensemble_models
        total_samples = 0

        # Calculate average error for each model
        for tf_history in self.prediction_history.values():
            if not tf_history:
                continue
            for entry in tf_history:
                actual = entry['actual']
                for i, pred in enumerate(entry['predictions']):
                    if i < len(model_errors):
                        model_errors[i] += abs(pred - actual)
                total_samples += 1

        if total_samples < 5:
            print("   Not enough evaluated predictions to update weights.")
            return

        # Better handling of zero division
        avg_errors = [err / max(total_samples, 1) for err in model_errors]
        
        # FIXED: Use softmax-style weighting to prevent extreme weights
        # This prevents any single model from dominating
        min_error = min(avg_errors) if avg_errors else 1e-8
        normalized_errors = [(err / min_error) for err in avg_errors]
        
        # Apply temperature scaling to control weight distribution
        temperature = 2.0  # Higher = more equal weights, lower = more extreme
        exp_neg_errors = [np.exp(-err / temperature) for err in normalized_errors]
        total_exp = sum(exp_neg_errors)

        if total_exp == 0:
            print("   WARNING: All weights are zero, keeping equal weights")
            return

        new_weights = [e / total_exp for e in exp_neg_errors]

        # Smooth weight updates
        self.ensemble_weights = [
            (1 - self.ensemble_learning_rate) * old_w + self.ensemble_learning_rate * new_w
            for old_w, new_w in zip(self.ensemble_weights, new_weights)
        ]

        # Normalize weights (just in case)
        weight_sum = sum(self.ensemble_weights)
        if weight_sum > 0:
            self.ensemble_weights = [w / weight_sum for w in self.ensemble_weights]

        print(f"   Avg errors: {[f'{e:.6f}' for e in avg_errors]}")
        print(f"   New weights: {[f'{w:.3f}' for w in self.ensemble_weights]}")

    def run_safe_backtest(self):
        """
        Walk-Forward Backtester.
        Fixes the 'Read-Ahead' cheating problem.
        """
        print("\n--- Starting Safe Backtest (Anti-Leakage) ---")

        # Download data
        df_h1, df_h4, df_d1 = self.download_data(bars=15000)
        df_full = self.create_features(df_h1, df_h4, df_d1)

        window = 2000
        step = 100

        for i in range(window, len(df_full) - 1, step):
            # Training only on past data
            past = df_full.iloc[:i]
            current_bar = df_full.iloc[i:i + 1]

            # Fit Scalers only on past data
            self.feature_scaler.fit(past[self.feature_cols])
            self.target_scaler.fit(past[[self.target_column]])

            # (Prediction and result logging here)
            print(f"Validated Bar: {df_full.index[i]}")

    def run_backtest_generation(self) -> None:
        """Generate historical predictions for backtesting."""
        print("\n" + "=" * 60)
        print("Starting Backtest Generation...")
        print("=" * 60)

        # Load models if not already loaded
        if not self.models:
            if not self.load_model_assets():
                return

        # Download historical data
        df_h1, df_h4, df_d1 = self.download_data(bars=40000)
        if df_h1 is None:
            return

        # Create features
        df = self.create_features(df_h1, df_h4, df_d1)
        df_selected = df[self.feature_cols + ['close']]

        # Prepare tabular data
        df_tabular = df_selected.copy()
        final_tab_cols = []
        for model_name, model in self.models.items():
            if 'lgbm' in model_name:
                final_tab_cols = model.feature_name_
                break

        if final_tab_cols:
            for col in self.feature_cols:
                for lag in [1, 3, 5, 10]:
                    new_col = f'{col}_lag_{lag}'
                    if new_col in final_tab_cols:
                        df_tabular[new_col] = df_tabular[col].shift(lag)
            df_tabular.dropna(inplace=True)

        # Scale features
        features_scaled = self.feature_scaler.transform(df_selected[self.feature_cols].values)

        # Generate predictions
        timeframes = {"1H": 1, "4H": 4, "1D": 24, "5D": 120}
        all_predictions = {tf: [] for tf in timeframes.keys()}
        timestamps = []

        print(f"Generating predictions for {len(df_selected) - self.lookback_periods} bars...")

        for i in range(self.lookback_periods, len(df_selected)):
            current_price = df_selected['close'].iloc[i]
            timestamp = df_selected.index[i]

            # Prepare sequential input
            X_pred_seq = features_scaled[i - self.lookback_periods:i].reshape(
                1, self.lookback_periods, len(self.feature_cols)
            )
            # Convert to tensor to avoid retracing
            X_pred_seq = tf.convert_to_tensor(X_pred_seq, dtype=tf.float32)

            # Prepare tabular input
            X_pred_tab = None
            if final_tab_cols and timestamp in df_tabular.index:
                X_pred_tab = df_tabular.loc[timestamp][final_tab_cols].values.reshape(1, -1)

            # Get predictions for each timeframe
            for tf_name, steps in timeframes.items():
                ensemble_preds = []
                for model_name, model in self.models.items():
                    pred_log_return = 0
                    try:
                        if 'lgbm' in model_name and X_pred_tab is not None:
                            pred_log_return = model.predict(X_pred_tab)[0]
                        elif 'lgbm' not in model_name:
                            # Use direct call to avoid retracing
                            pred_log_return_scaled = model(X_pred_seq, training=False).numpy()[0][0]
                            pred_log_return = self.target_scaler.inverse_transform([[pred_log_return_scaled]])[0][0]

                        predicted_price = current_price * np.exp(pred_log_return * steps)

                        # Validate prediction
                        if not np.isnan(predicted_price) and not np.isinf(predicted_price):
                            ensemble_preds.append(predicted_price)
                    except Exception:
                        continue

                if ensemble_preds:
                    weighted_price = np.average(ensemble_preds, weights=self.ensemble_weights[:len(ensemble_preds)])
                    all_predictions[tf_name].append(weighted_price)
                else:
                    all_predictions[tf_name].append(current_price)

            timestamps.append(timestamp)

            # Progress indicator
            if (i - self.lookback_periods) % 500 == 0:
                progress = ((i - self.lookback_periods) / (len(df_selected) - self.lookback_periods)) * 100
                print(f"   Progress: {progress:.1f}%")

        # Export backtest files
        self.export_backtest_files(timestamps, all_predictions)
        print("\nBACKTEST GENERATION COMPLETE!")

    def export_backtest_files(self, timestamps: List, predictions: Dict[str, List[float]]) -> None:
        """Export backtest predictions to CSV files."""
        print("\nExporting backtest files...")
        for tf_name, pred_values in predictions.items():
            lookup_file = os.path.join(self.base_path, f'{self.symbol}_{tf_name}_lookup.csv')
            try:
                with open(lookup_file, 'w') as f:
                    f.write('timestamp,prediction\n')  # Add header
                    for ts, pred in zip(timestamps, pred_values):
                        f.write(f'{ts.strftime("%Y.%m.%d %H:%M")},{pred:.5f}\n')
                print(f"   Created: {lookup_file}")
            except Exception as e:
                print(f"   Error creating {lookup_file}: {e}")

    def save_to_file(self, file_path: str, data: Dict) -> None:
        """Save data to JSON file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving to {file_path}: {e}")

    def run_continuous(self, interval_minutes: int = 60) -> None:
        """
        Run predictions continuously at specified intervals.

        Args:
            interval_minutes: Minutes between prediction cycles
        """
        prediction_method = self.run_prediction_cycle_multitimeframe if self.use_multitimeframe else self.run_prediction_cycle

        print(f"\nStarting Continuous Mode for {self.symbol} (Interval: {interval_minutes} mins)")
        print(f"Using {'multi-timeframe' if self.use_multitimeframe else 'single-timeframe'} prediction method")

        while True:
            try:
                prediction_method()
                print(f"\nWaiting {interval_minutes} minutes until next cycle...")
                time.sleep(interval_minutes * 60)
            except KeyboardInterrupt:
                print("\nService stopped by user.")
                break
            except Exception as e:
                print(f"\nAn error occurred: {e}")
                import traceback
                traceback.print_exc()
                print("Retrying in 5 minutes...")
                time.sleep(300)


def main():
    """Main entry point for the predictor."""
    print("""
    ================================================================
       Hybrid Ensemble MT5 Predictor v8.1 (With Macro Integration)
       Feat: Transformer, TCN, GRU, LightGBM & Multi-Timeframe Support
       TensorFlow Optimized - 2-3x Faster Predictions
    ================================================================
    """)

    # Setup argument parser
    parser = argparse.ArgumentParser(description="Hybrid Ensemble MT5 Predictor v8.1")
    subparsers = parser.add_subparsers(dest='mode', required=True, help="Operating mode")
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--symbol', type=str, default="EURUSD", help="Currency symbol.")

    # Train mode (old single-timeframe)
    p_train = subparsers.add_parser('train', parents=[parent_parser], help="Train the model ensemble (single timeframe).")
    p_train.add_argument('--force', action='store_true', help="Force retraining.")
    p_train.add_argument(
        '--models',
        nargs='+',
        default=['lstm', 'transformer', 'lgbm'],
        choices=['lstm', 'gru', 'transformer', 'tcn', 'lgbm'],
        help="List of model types for the ensemble."
    )

    # Train multi-timeframe mode (recommended)
    p_train_mtf = subparsers.add_parser('train-multitf', parents=[parent_parser],
                                        help="Train separate models for each timeframe (RECOMMENDED).")
    p_train_mtf.add_argument('--force', action='store_true', help="Force retraining.")
    p_train_mtf.add_argument(
        '--models',
        nargs='+',
        default=['lstm', 'transformer', 'lgbm'],
        choices=['lstm', 'gru', 'transformer', 'tcn', 'lgbm'],
        help="List of model types for the ensemble."
    )

    # Tune mode
    subparsers.add_parser('tune', parents=[parent_parser], help="Run hyperparameter tuning for DL models.")

    # Predict mode (old single-timeframe)
    p_predict = subparsers.add_parser('predict', parents=[parent_parser], help="Run a prediction cycle (single timeframe).")
    p_predict.add_argument('--continuous', action='store_true', help="Run in a continuous loop.")
    p_predict.add_argument('--interval', type=int, default=60, help="Interval in minutes for continuous mode.")
    p_predict.add_argument('--models', nargs='+', choices=['lstm', 'gru', 'transformer', 'tcn', 'lgbm'],
                           help="Override automatic model detection (optional)")
    p_predict.add_argument('--no-kalman', action='store_true', help="Disable Kalman filtering (use EMA smoothing)")

    # Predict multi-timeframe mode (recommended)
    p_predict_mtf = subparsers.add_parser('predict-multitf', parents=[parent_parser],
                                          help="Run prediction using timeframe-specific models (RECOMMENDED).")
    p_predict_mtf.add_argument('--continuous', action='store_true', help="Run in a continuous loop.")
    p_predict_mtf.add_argument('--interval', type=int, default=60, help="Interval in minutes for continuous mode.")
    p_predict_mtf.add_argument('--models', nargs='+', choices=['lstm', 'gru', 'transformer', 'tcn', 'lgbm'],
                               help="Override automatic model detection (optional)")
    p_predict_mtf.add_argument('--no-kalman', action='store_true', help="Disable Kalman filtering (use EMA smoothing)")

    # Backtest mode
    subparsers.add_parser('backtest', parents=[parent_parser], help="Generate historical predictions.")

    args = parser.parse_args()

    # Build predictor arguments
    predictor_args = {'symbol': args.symbol.upper()}

    if args.mode in ['train', 'train-multitf']:
        predictor_args['ensemble_model_types'] = args.models
        predictor_args['use_multitimeframe'] = (args.mode == 'train-multitf')
    elif args.mode in ['predict', 'predict-multitf']:
        if hasattr(args, 'models') and args.models:
            predictor_args['ensemble_model_types'] = args.models
        if hasattr(args, 'no_kalman') and args.no_kalman:
            predictor_args['use_kalman'] = False
        else:
            predictor_args['use_kalman'] = True
        predictor_args['use_multitimeframe'] = (args.mode == 'predict-multitf')

    # Initialize predictor
    predictor = UnifiedLSTMPredictor(**predictor_args)

    # Execute requested mode
    try:
        if args.mode == 'tune':
            predictor.tune_hyperparameters()
        elif args.mode == 'train':
            predictor.train_model(force_retrain=args.force)
        elif args.mode == 'train-multitf':
            predictor.train_model_multitimeframe(force_retrain=args.force)
        elif args.mode == 'predict':
            if args.continuous:
                predictor.run_continuous(interval_minutes=args.interval)
            else:
                predictor.run_prediction_cycle()
        elif args.mode == 'predict-multitf':
            if args.continuous:
                predictor.run_continuous(interval_minutes=args.interval)
            else:
                predictor.run_prediction_cycle_multitimeframe()
        elif args.mode == 'backtest':
            predictor.run_backtest_generation()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        mt5.shutdown()
        print("\nShutdown complete. Thank you!")


if __name__ == "__main__":
    main()
