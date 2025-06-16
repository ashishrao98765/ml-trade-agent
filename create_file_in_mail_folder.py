import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SRLevel:
    """Support/Resistance level class matching Pine Script logic"""
    def __init__(self, price, touches=1, is_valid=False, first_touch_bar=0, 
                 last_touch_bar=0, level_color='blue'):
        self.price = price
        self.touches = touches
        self.is_valid = is_valid
        self.first_touch_bar = first_touch_bar
        self.last_touch_bar = last_touch_bar
        self.level_color = level_color

class AdvancedTradingSystemML:
    def __init__(self, symbol, period='2y', lookback=14, mult=1.0, 
                 calc_method='atr', min_touches=2, touch_tolerance=0.2):
        self.symbol = symbol
        self.period = period
        self.lookback = lookback
        self.mult = mult
        self.calc_method = calc_method
        self.min_touches = min_touches
        self.touch_tolerance = touch_tolerance
        
        # Data storage
        self.data = None
        self.ml_features = None
        self.signals = pd.DataFrame()
        
        # ML components
        self.ml_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # S/R levels (matching Pine Script)
        self.resistance_levels = []
        self.support_levels = []
        
        # Pivot lengths (matching Pine Script)
        self.length1 = 10  # Short term
        self.length2 = 20  # Medium term  
        self.length3 = 40  # Long term
        self.length4 = 60  # Major
        
        # Strategy settings
        self.enable_long = True
        self.enable_short = True
        self.use_sr_exits = True
        self.sr_exit_sensitivity = 0.1
        
    def fetch_data(self):
        """Fetch stock data using yfinance"""
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period)
            if len(self.data) < 100:
                print("Warning: Limited data available")
                return False
            print(f"Data fetched for {self.symbol}: {len(self.data)} rows")
            return True
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
    
    def calculate_trendlines(self):
        """Calculate trendlines exactly like Pine Script"""
        df = self.data.copy()
        n = len(df)
        
        # Calculate pivot highs and lows
        df['pivot_high'] = df['High'].rolling(window=self.lookback*2+1, center=True).max() == df['High']
        df['pivot_low'] = df['Low'].rolling(window=self.lookback*2+1, center=True).min() == df['Low']
        
        # Slope calculation (matching Pine Script methods)
        if self.calc_method == 'atr':
            df['atr'] = self.calculate_atr(df, self.lookback)
            slope = df['atr'] / self.lookback * self.mult
        elif self.calc_method == 'stdev':
            slope = df['Close'].rolling(self.lookback).std() / self.lookback * self.mult
        else:  # linreg
            slope = self.calculate_linreg_slope(df) * self.mult
        
        # Initialize trendline variables
        upper = np.full(len(df), np.nan)
        lower = np.full(len(df), np.nan)
        slope_ph = np.full(len(df), np.nan)
        slope_pl = np.full(len(df), np.nan)
        
        # Calculate trendlines (Pine Script logic)
        for i in range(self.lookback, len(df)):
            # Update slopes on pivot points
            if df['pivot_high'].iloc[i]:
                slope_ph[i] = slope.iloc[i]
                upper[i] = df['High'].iloc[i]
            else:
                slope_ph[i] = slope_ph[i-1] if not np.isnan(slope_ph[i-1]) else slope.iloc[i]
                upper[i] = upper[i-1] - slope_ph[i] if not np.isnan(upper[i-1]) else df['High'].iloc[i]
            
            if df['pivot_low'].iloc[i]:
                slope_pl[i] = slope.iloc[i]
                lower[i] = df['Low'].iloc[i]
            else:
                slope_pl[i] = slope_pl[i-1] if not np.isnan(slope_pl[i-1]) else slope.iloc[i]
                lower[i] = lower[i-1] + slope_pl[i] if not np.isnan(lower[i-1]) else df['Low'].iloc[i]
        
        df['upper_trendline'] = upper
        df['lower_trendline'] = lower
        df['slope_ph'] = slope_ph
        df['slope_pl'] = slope_pl
        
        return df
    
    def calculate_atr(self, df, period):
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        return tr.rolling(window=period).mean()
    
    def calculate_linreg_slope(self, df):
        """Calculate linear regression slope"""
        slopes = []
        for i in range(len(df)):
            if i < self.lookback:
                slopes.append(0)
            else:
                y = df['Close'].iloc[i-self.lookback:i].values
                x = np.arange(len(y))
                slope = np.polyfit(x, y, 1)[0]
                slopes.append(abs(slope))
        return pd.Series(slopes, index=df.index)
    
    def detect_sr_levels(self, df):
        """Detect and validate S/R levels (Pine Script logic)"""
        # Clear existing levels
        self.resistance_levels = []
        self.support_levels = []
        
        # Calculate pivots for different timeframes
        pivots = {}
        for length in [self.length1, self.length2, self.length3, self.length4]:
            pivots[f'ph_{length}'] = df['High'].rolling(window=length*2+1, center=True).max() == df['High']
            pivots[f'pl_{length}'] = df['Low'].rolling(window=length*2+1, center=True).min() == df['Low']
        
        # Process each pivot point
        for i in range(max(self.length1, self.length2, self.length3, self.length4), len(df)):
            current_high = df['High'].iloc[i]
            current_low = df['Low'].iloc[i]
            
            # Check for new resistance levels
            for length in [self.length1, self.length2, self.length3, self.length4]:
                if pivots[f'ph_{length}'].iloc[i]:
                    self.add_resistance_level(current_high, i)
                if pivots[f'pl_{length}'].iloc[i]:
                    self.add_support_level(current_low, i)
            
            # Update touch counts for existing levels
            self.update_level_touches(df.iloc[i], i)
    
    def add_resistance_level(self, price, bar_index):
        """Add or update resistance level"""
        # Check if level already exists
        for level in self.resistance_levels:
            if abs(level.price - price) < price * (self.touch_tolerance / 100):
                level.touches += 1
                level.last_touch_bar = bar_index
                level.is_valid = level.touches >= self.min_touches
                return
        
        # Add new level
        new_level = SRLevel(price, 1, False, bar_index, bar_index, 'red')
        self.resistance_levels.append(new_level)
    
    def add_support_level(self, price, bar_index):
        """Add or update support level"""
        # Check if level already exists
        for level in self.support_levels:
            if abs(level.price - price) < price * (self.touch_tolerance / 100):
                level.touches += 1
                level.last_touch_bar = bar_index
                level.is_valid = level.touches >= self.min_touches
                return
        
        # Add new level
        new_level = SRLevel(price, 1, False, bar_index, bar_index, 'green')
        self.support_levels.append(new_level)
    
    def update_level_touches(self, current_bar, bar_index):
        """Update touch counts for existing levels"""
        high_price = current_bar['High']
        low_price = current_bar['Low']
        
        # Check resistance levels
        for level in self.resistance_levels:
            if (bar_index > level.last_touch_bar + 1 and
                high_price >= level.price * (1 - self.touch_tolerance / 100) and
                high_price <= level.price * (1 + self.touch_tolerance / 100)):
                level.touches += 1
                level.last_touch_bar = bar_index
                level.is_valid = level.touches >= self.min_touches
        
        # Check support levels  
        for level in self.support_levels:
            if (bar_index > level.last_touch_bar + 1 and
                low_price <= level.price * (1 + self.touch_tolerance / 100) and
                low_price >= level.price * (1 - self.touch_tolerance / 100)):
                level.touches += 1
                level.last_touch_bar = bar_index
                level.is_valid = level.touches >= self.min_touches
    
    def calculate_volume_indicators(self, df):
        """Calculate volume-based indicators"""
        # Volume Moving Average
        df['volume_ma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma']
        
        # Volume Profile (simplified)
        df['price_volume'] = df['Close'] * df['Volume']
        df['vwap'] = df['price_volume'].rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
        
        # Volume Spike Detection
        df['volume_spike'] = df['volume_ratio'] > 1.5
        
        # On Balance Volume
        df['obv'] = 0.0
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                df.loc[df.index[i], 'obv'] = df['obv'].iloc[i-1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                df.loc[df.index[i], 'obv'] = df['obv'].iloc[i-1] - df['Volume'].iloc[i]
            else:
                df.loc[df.index[i], 'obv'] = df['obv'].iloc[i-1]
        
        return df
    
    def calculate_stochastic(self, df, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        df['lowest_low'] = df['Low'].rolling(window=k_period).min()
        df['highest_high'] = df['High'].rolling(window=k_period).max()
        
        df['stoch_k'] = 100 * ((df['Close'] - df['lowest_low']) / 
                              (df['highest_high'] - df['lowest_low']))
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        
        # Stochastic signals
        df['stoch_oversold'] = df['stoch_k'] < 20
        df['stoch_overbought'] = df['stoch_k'] > 80
        df['stoch_bullish_cross'] = (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))
        df['stoch_bearish_cross'] = (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))
        
        return df
    
    def create_ml_features(self, df):
        """Create comprehensive feature set for ML model"""
        features_df = df.copy()
        
        # Price-based features
        features_df['returns'] = df['Close'].pct_change()
        features_df['volatility'] = features_df['returns'].rolling(window=10).std()
        features_df['rsi'] = self.calculate_rsi(df['Close'])
        
        # Trendline features
        features_df['price_above_upper'] = df['Close'] > df['upper_trendline']
        features_df['price_below_lower'] = df['Close'] < df['lower_trendline']
        features_df['distance_to_upper'] = (df['upper_trendline'] - df['Close']) / df['Close']
        features_df['distance_to_lower'] = (df['Close'] - df['lower_trendline']) / df['Close']
        
        # S/R features
        features_df['near_resistance'] = 0
        features_df['near_support'] = 0
        features_df['valid_resistance_count'] = 0
        features_df['valid_support_count'] = 0
        
        # Volume features (already calculated)
        # Stochastic features (already calculated)
        
        # Market structure features
        features_df['higher_high'] = (df['High'] > df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2))
        features_df['lower_low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2))
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features_df[f'ma_{period}'] = df['Close'].rolling(window=period).mean()
            features_df[f'price_above_ma_{period}'] = df['Close'] > features_df[f'ma_{period}']
        
        return features_df
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self, df):
        """Generate trading signals (Pine Script logic)"""
        signals = pd.DataFrame(index=df.index)
        signals['long_signal'] = False
        signals['short_signal'] = False
        signals['long_exit'] = False
        signals['short_exit'] = False
        
        # Trendline breakout signals (Pine Script logic)
        upos = np.zeros(len(df))
        dnos = np.zeros(len(df))
        
        for i in range(1, len(df)):
            # Upward position tracking
            if not pd.isna(df['upper_trendline'].iloc[i-1]):
                if df['Close'].iloc[i] > (df['upper_trendline'].iloc[i] - df['slope_ph'].iloc[i] * self.lookback):
                    upos[i] = 1
                else:
                    upos[i] = upos[i-1]
            
            # Downward position tracking  
            if not pd.isna(df['lower_trendline'].iloc[i-1]):
                if df['Close'].iloc[i] < (df['lower_trendline'].iloc[i] + df['slope_pl'].iloc[i] * self.lookback):
                    dnos[i] = 1
                else:
                    dnos[i] = dnos[i-1]
            
            # Signal generation
            upward_break = upos[i] > upos[i-1]
            downward_break = dnos[i] > dnos[i-1]
            
            # Additional confirmations
            volume_confirm = df['volume_ratio'].iloc[i] > 1.2
            stoch_confirm_long = not df['stoch_overbought'].iloc[i]
            stoch_confirm_short = not df['stoch_oversold'].iloc[i]
            
            # Enhanced signals with confirmations
            if upward_break and volume_confirm and stoch_confirm_long:
                signals.loc[signals.index[i], 'long_signal'] = True
            
            if downward_break and volume_confirm and stoch_confirm_short:
                signals.loc[signals.index[i], 'short_signal'] = True
        
        df['upos'] = upos
        df['dnos'] = dnos
        
        return signals
    
    def prepare_ml_data(self, df, signals):
        """Prepare data for ML training"""
        # Create target variable (future returns)
        df['future_return'] = df['Close'].shift(-5) / df['Close'] - 1
        df['target'] = np.where(df['future_return'] > 0.01, 1, 
                               np.where(df['future_return'] < -0.01, -1, 0))
        
        # Select features for ML
        feature_columns = [
            'returns', 'volatility', 'rsi', 'volume_ratio', 'volume_spike',
            'stoch_k', 'stoch_d', 'stoch_oversold', 'stoch_overbought',
            'price_above_upper', 'price_below_lower', 'distance_to_upper', 'distance_to_lower',
            'higher_high', 'lower_low', 'price_above_ma_5', 'price_above_ma_10',
            'price_above_ma_20', 'price_above_ma_50'
        ]
        
        # Create feature matrix
        X = df[feature_columns].dropna()
        y = df['target'].loc[X.index]
        
        return X, y
    
    def train_ml_model(self, X, y):
        """Train ML model for signal enhancement"""
        # Remove neutral signals for training
        mask = y != 0
        X_filtered = X[mask]
        y_filtered = y[mask]
        
        if len(X_filtered) < 50:
            print("Insufficient data for ML training")
            return False
        
        # Split data chronologically
        split_point = int(len(X_filtered) * 0.8)
        X_train = X_filtered.iloc[:split_point]
        X_test = X_filtered.iloc[split_point:]
        y_train = y_filtered.iloc[:split_point]
        y_test = y_filtered.iloc[split_point:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.ml_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.ml_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.ml_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"ML Model Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.is_trained = True
        return True
    
    def get_ml_prediction(self, features):
        """Get ML prediction for current market state"""
        if not self.is_trained or self.ml_model is None:
            return 0
        
        try:
            features_scaled = self.scaler.transform(features.values.reshape(1, -1))
            prediction = self.ml_model.predict(features_scaled)[0]
            probabilities = self.ml_model.predict_proba(features_scaled)[0]
            confidence = max(probabilities)
            
            return prediction if confidence > 0.6 else 0
        except:
            return 0
    
    def run_complete_analysis(self):
        """Run complete trading system analysis"""
        print(f"Starting analysis for {self.symbol}...")
        
        # 1. Fetch data
        if not self.fetch_data():
            return None
        
        # 2. Calculate trendlines
        print("Calculating trendlines...")
        df = self.calculate_trendlines()
        
        # 3. Detect S/R levels
        print("Detecting S/R levels...")
        self.detect_sr_levels(df)
        
        # 4. Calculate volume indicators
        print("Calculating volume indicators...")
        df = self.calculate_volume_indicators(df)
        
        # 5. Calculate stochastic
        print("Calculating stochastic indicators...")
        df = self.calculate_stochastic(df)
        
        # 6. Create ML features
        print("Creating ML features...")
        df = self.create_ml_features(df)
        
        # 7. Generate initial signals
        print("Generating signals...")
        signals = self.generate_signals(df)
        
        # 8. Prepare ML data and train
        print("Training ML model...")
        X, y = self.prepare_ml_data(df, signals)
        self.train_ml_model(X, y)
        
        # 9. Enhanced signals with ML
        enhanced_signals = self.generate_enhanced_signals(df, signals)
        
        # 10. Backtest results
        results = self.backtest_strategy(df, enhanced_signals)
        
        self.data = df
        self.signals = enhanced_signals
        
        return results
    
    def generate_enhanced_signals(self, df, original_signals):
        """Generate enhanced signals using ML"""
        enhanced_signals = original_signals.copy()
        
        if not self.is_trained:
            return enhanced_signals
        
        feature_columns = [
            'returns', 'volatility', 'rsi', 'volume_ratio', 'volume_spike',
            'stoch_k', 'stoch_d', 'stoch_oversold', 'stoch_overbought',
            'price_above_upper', 'price_below_lower', 'distance_to_upper', 'distance_to_lower',
            'higher_high', 'lower_low', 'price_above_ma_5', 'price_above_ma_10',
            'price_above_ma_20', 'price_above_ma_50'
        ]
        
        for i in range(len(df)):
            if i < 50:  # Skip early bars
                continue
                
            try:
                features = df[feature_columns].iloc[i]
                if features.isna().any():
                    continue
                    
                ml_prediction = self.get_ml_prediction(features)
                
                # Enhance original signals with ML
                if original_signals['long_signal'].iloc[i] and ml_prediction == 1:
                    enhanced_signals.loc[enhanced_signals.index[i], 'long_signal'] = True
                elif original_signals['short_signal'].iloc[i] and ml_prediction == -1:
                    enhanced_signals.loc[enhanced_signals.index[i], 'short_signal'] = True
                else:
                    enhanced_signals.loc[enhanced_signals.index[i], 'long_signal'] = False
                    enhanced_signals.loc[enhanced_signals.index[i], 'short_signal'] = False
                    
            except Exception as e:
                continue
        
        return enhanced_signals
    
    def backtest_strategy(self, df, signals):
        """Backtest the strategy"""
        portfolio_value = 10000
        position = 0
        trades = []
        equity_curve = []
        
        for i in range(len(df)):
            current_price = df['Close'].iloc[i]
            date = df.index[i]
            
            # Entry signals
            if signals['long_signal'].iloc[i] and position <= 0:
                if position < 0:  # Close short
                    pnl = (position * current_price) - (position * entry_price)
                    portfolio_value += pnl
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'type': 'short',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': pnl
                    })
                
                # Open long
                position = portfolio_value / current_price
                entry_price = current_price
                entry_date = date
                
            elif signals['short_signal'].iloc[i] and position >= 0:
                if position > 0:  # Close long
                    pnl = (position * current_price) - (position * entry_price)
                    portfolio_value += pnl
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'type': 'long',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': pnl
                    })
                
                # Open short
                position = -portfolio_value / current_price
                entry_price = current_price
                entry_date = date
            
            # Calculate current portfolio value
            if position > 0:
                current_value = position * current_price
            elif position < 0:
                current_value = portfolio_value - (abs(position) * current_price - abs(position) * entry_price)
            else:
                current_value = portfolio_value
                
            equity_curve.append(current_value)
        
        # Calculate performance metrics
        equity_series = pd.Series(equity_curve, index=df.index)
        returns = equity_series.pct_change().dropna()
        
        total_return = (equity_curve[-1] / 10000 - 1) * 100
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        max_drawdown = ((equity_series / equity_series.expanding().max()) - 1).min() * 100
        
        win_trades = [t for t in trades if t['pnl'] > 0]
        win_rate = len(win_trades) / len(trades) * 100 if trades else 0
        
        results = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'equity_curve': equity_series,
            'trades': trades
        }
        
        return results
    
    def plot_analysis(self):
        """Plot comprehensive analysis"""
        if self.data is None:
            print("No data available. Run analysis first.")
            return

        fig, axes = plt.subplots(4, 1, figsize=(15, 20))

        # 1. Price with trendlines and S/R levels
        self._plot_price_trendlines_sr(axes[0])

        # 2. Volume analysis
        self._plot_volume_analysis(axes[1])

        # 3. Stochastic Oscillator
        ax3 = axes[2]
        ax3.plot(self.data.index, self.data['stoch_k'], label='%K', color='blue')
        ax3.plot(self.data.index, self.data['stoch_d'], label='%D', color='red')
        ax3.axhline(y=80, color='red', linestyle='--', alpha=0.7)
        ax3.axhline(y=20, color='green', linestyle='--', alpha=0.7)
        ax3.fill_between(self.data.index, 80, 100, alpha=0.2, color='red')
        ax3.fill_between(self.data.index, 0, 20, alpha=0.2, color='green')
        ax3.set_title('Stochastic Oscillator')
        ax3.set_ylim(0, 100)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. RSI and additional indicators
        ax4 = axes[3]
        ax4.plot(self.data.index, self.data['rsi'], label='RSI', color='purple')
        ax4.axhline(y=70, color='red', linestyle='--', alpha=0.7)
        ax4.axhline(y=30, color='green', linestyle='--', alpha=0.7)
        ax4.fill_between(self.data.index, 70, 100, alpha=0.2, color='red')
        ax4.fill_between(self.data.index, 0, 30, alpha=0.2, color='green')
        ax4.set_title('RSI Indicator')
        ax4.set_ylim(0, 100)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _plot_price_trendlines_sr(self, ax):
        """Helper to plot price, trendlines, and S/R levels"""
        ax.plot(self.data.index, self.data['Close'], label='Close Price', linewidth=1)
        ax.plot(self.data.index, self.data['upper_trendline'], 'r--', label='Upper Trendline', alpha=0.7)
        ax.plot(self.data.index, self.data['lower_trendline'], 'g--', label='Lower Trendline', alpha=0.7)

        # Plot S/R levels
        for level in self.resistance_levels:
            if level.is_valid:
                ax.axhline(y=level.price, color='red', linestyle='-', alpha=0.5, linewidth=2)

        for level in self.support_levels:
            if level.is_valid:
                ax.axhline(y=level.price, color='green', linestyle='-', alpha=0.5, linewidth=2)

        # Plot signals
        long_signals = self.signals[self.signals['long_signal']]
        short_signals = self.signals[self.signals['short_signal']]

        ax.scatter(long_signals.index, self.data.loc[long_signals.index, 'Close'],
                   color='green', marker='^', s=100, label='Long Signal', zorder=5)
        ax.scatter(short_signals.index, self.data.loc[short_signals.index, 'Close'],
                   color='red', marker='v', s=100, label='Short Signal', zorder=5)

        ax.set_title(f'{self.symbol} - Price with Trendlines and S/R Levels')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_volume_analysis(self, ax):
        """Helper to plot volume analysis"""
        ax.bar(self.data.index, self.data['Volume'], alpha=0.6, color='blue')
        ax.plot(self.data.index, self.data['volume_ma'], color='orange', label='Volume MA')
        ax.set_title('Volume Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_backtest_results(self, results):
        """Plot backtest results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Equity Curve
        ax1 = axes[0, 0]
        ax1.plot(results['equity_curve'].index, results['equity_curve'], 
                linewidth=2, color='blue')
        ax1.set_title('Portfolio Equity Curve')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown
        equity = results['equity_curve']
        rolling_max = equity.expanding().max()
        drawdown = (equity / rolling_max - 1) * 100
        
        ax2 = axes[0, 1]
        ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.7, color='red')
        ax2.set_title('Drawdown (%)')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Trade Distribution
        if results['trades']:
            pnls = [trade['pnl'] for trade in results['trades']]
            ax3 = axes[1, 0]
            ax3.hist(pnls, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax3.set_title('Trade P&L Distribution')
            ax3.set_xlabel('P&L ($)')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
        
        # 4. Performance Metrics
        ax4 = axes[1, 1]
        metrics = {
            'Total Return (%)': f"{results['total_return']:.2f}",
            'Sharpe Ratio': f"{results['sharpe_ratio']:.2f}",
            'Max Drawdown (%)': f"{results['max_drawdown']:.2f}",
            'Win Rate (%)': f"{results['win_rate']:.2f}",
            'Total Trades': f"{results['total_trades']}"
        }
        
        y_pos = range(len(metrics))
        ax4.barh(y_pos, [float(v.replace('%', '')) if '%' in v else float(v) for v in metrics.values()],
                alpha=0.7, color='lightblue')
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(metrics.keys())
        ax4.set_title('Performance Metrics')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_current_signals(self):
        """Get current trading signals"""
        if self.data is None or len(self.data) == 0:
            return None
        
        latest_idx = len(self.data) - 1
        current_price = self.data['Close'].iloc[latest_idx]
        
        # Check S/R levels proximity
        near_resistance = False
        near_support = False
        resistance_distance = float('inf')
        support_distance = float('inf')
        
        for level in self.resistance_levels:
            if level.is_valid:
                distance = abs(level.price - current_price) / current_price * 100
                if distance < 2:  # Within 2%
                    near_resistance = True
                    resistance_distance = min(resistance_distance, distance)
        
        for level in self.support_levels:
            if level.is_valid:
                distance = abs(level.price - current_price) / current_price * 100
                if distance < 2:  # Within 2%
                    near_support = True
                    support_distance = min(support_distance, distance)
        
        # Get ML prediction if available
        ml_signal = 0
        if self.is_trained and latest_idx >= 50:
            feature_columns = [
                'returns', 'volatility', 'rsi', 'volume_ratio', 'volume_spike',
                'stoch_k', 'stoch_d', 'stoch_oversold', 'stoch_overbought',
                'price_above_upper', 'price_below_lower', 'distance_to_upper', 'distance_to_lower',
                'higher_high', 'lower_low', 'price_above_ma_5', 'price_above_ma_10',
                'price_above_ma_20', 'price_above_ma_50'
            ]
            
            try:
                features = self.data[feature_columns].iloc[latest_idx]
                if not features.isna().any():
                    ml_signal = self.get_ml_prediction(features)
            except:
                pass
        
        return {
            'symbol': self.symbol,
            'current_price': current_price,
            'timestamp': self.data.index[latest_idx],
            'long_signal': self.signals['long_signal'].iloc[latest_idx] if len(self.signals) > latest_idx else False,
            'short_signal': self.signals['short_signal'].iloc[latest_idx] if len(self.signals) > latest_idx else False,
            'ml_signal': ml_signal,
            'near_resistance': near_resistance,
            'near_support': near_support,
            'resistance_distance': resistance_distance if resistance_distance != float('inf') else None,
            'support_distance': support_distance if support_distance != float('inf') else None,
            'rsi': self.data['rsi'].iloc[latest_idx],
            'stoch_k': self.data['stoch_k'].iloc[latest_idx],
            'volume_ratio': self.data['volume_ratio'].iloc[latest_idx],
            'trend_position': 'Above Upper' if self.data['price_above_upper'].iloc[latest_idx] else 
                           'Below Lower' if self.data['price_below_lower'].iloc[latest_idx] else 'Between'
        }
    
    def print_performance_summary(self, results):
        """Print detailed performance summary"""
        print("\n" + "="*60)
        print(f"TRADING SYSTEM PERFORMANCE SUMMARY - {self.symbol}")
        print("="*60)
        
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"Maximum Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Win Rate: {results['win_rate']:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        
        if results['trades']:
            winning_trades = [t for t in results['trades'] if t['pnl'] > 0]
            losing_trades = [t for t in results['trades'] if t['pnl'] < 0]
            
            if winning_trades:
                avg_win = np.mean([t['pnl'] for t in winning_trades])
                print(f"Average Winning Trade: ${avg_win:.2f}")
            
            if losing_trades:
                avg_loss = np.mean([t['pnl'] for t in losing_trades])
                print(f"Average Losing Trade: ${avg_loss:.2f}")
            
            if winning_trades and losing_trades:
                profit_factor = abs(sum(t['pnl'] for t in winning_trades)) / abs(sum(t['pnl'] for t in losing_trades))
                print(f"Profit Factor: {profit_factor:.3f}")
        
        print("\nSupport/Resistance Levels Found:")
        print(f"Valid Resistance Levels: {len([l for l in self.resistance_levels if l.is_valid])}")
        print(f"Valid Support Levels: {len([l for l in self.support_levels if l.is_valid])}")
        
        print("\n" + "="*60)


# Example usage and testing function
def run_trading_system_example():
    """Example of how to use the trading system"""
    
    # Initialize the trading system
    symbol = "AAPL"  # You can change this to any stock symbol
    system = AdvancedTradingSystemML(
        symbol=symbol,
        period='2y',
        lookback=14,
        mult=1.0,
        calc_method='atr',
        min_touches=2,
        touch_tolerance=0.2
    )
    
    print(f"Initializing Advanced Trading System for {symbol}...")
    
    # Run complete analysis
    results = system.run_complete_analysis()
    
    if results is not None:
        # Print performance summary
        system.print_performance_summary(results)
        
        # Plot analysis
        system.plot_analysis()
        
        # Plot backtest results
        system.plot_backtest_results(results)
        
        # Get current signals
        current_signals = system.get_current_signals()
        if current_signals:
            print("\n" + "="*40)
            print("CURRENT MARKET ANALYSIS")
            print("="*40)
            for key, value in current_signals.items():
                if value is not None:
                    print(f"{key}: {value}")
        
        return system, results
    else:
        print("Failed to run analysis. Please check your internet connection and symbol.")
        return None, None


# Additional utility function for batch analysis
def analyze_multiple_symbols(symbols, period='1y'):
    """Analyze multiple symbols and compare results"""
    results_summary = []
    
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"Analyzing {symbol}...")
        print('='*50)
        
        try:
            system = AdvancedTradingSystemML(symbol=symbol, period=period)
            results = system.run_complete_analysis()
            
            if results:
                results_summary.append({
                    'symbol': symbol,
                    'total_return': results['total_return'],
                    'sharpe_ratio': results['sharpe_ratio'],
                    'max_drawdown': results['max_drawdown'],
                    'win_rate': results['win_rate'],
                    'total_trades': results['total_trades']
                })
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            continue
    
    # Create comparison DataFrame
    if results_summary:
        comparison_df = pd.DataFrame(results_summary)
        comparison_df = comparison_df.sort_values('total_return', ascending=False)
        
        print("\n" + "="*80)
        print("MULTI-SYMBOL PERFORMANCE COMPARISON")
        print("="*80)
        print(comparison_df.to_string(index=False, float_format='%.2f'))
        
        return comparison_df
    
    return None


if __name__ == "__main__":
    # Run example
    system, results = run_trading_system_example()
    
    # Uncomment below to analyze multiple symbols
    # symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    # comparison = analyze_multiple_symbols(symbols)