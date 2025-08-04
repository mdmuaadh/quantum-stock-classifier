from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import time
import gc
import json
import uuid
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'quantum-stock-classifier-2024'

# Global variables to store models and data
trained_models = {}
training_history = []

def get_memory_usage():
    """Get current memory usage percentage (simplified for Railway)"""
    return 50.0  # Return a default value for Railway deployment

def load_data_for_training():
    """Load and prepare data with enhanced features (Railway-compatible)"""
    # Create mock data for Railway deployment (no large CSV files needed)
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic stock data
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META']
    data = []
    
    for ticker in tickers:
        for i in range(n_samples // len(tickers)):
            close_price = np.random.normal(150, 50)
            volume = np.random.lognormal(15, 1)
            data.append({
                'Close': close_price,
                'Volume': volume,
                'Ticker': ticker,
                'Label': np.random.choice([0, 1])  # Buy/Sell signal
            })
    
    df = pd.DataFrame(data)
    
    # Enhanced feature engineering
    feature_cols = ['Close', 'Volume']
    
    # Log transform volume
    df['Volume_log'] = np.log1p(df['Volume'])
    feature_cols.append('Volume_log')
    
    # Price volatility features
    if len(df) > 1000:
        df['Close_pct_change'] = df.groupby('Ticker')['Close'].pct_change().fillna(0)
        df['Price_volatility'] = df.groupby('Ticker')['Close_pct_change'].rolling(5, min_periods=1).std().reset_index(0, drop=True).fillna(0)
        feature_cols.extend(['Close_pct_change', 'Price_volatility'])
    
    # Ticker encoding
    if 'Ticker' in df.columns:
        ticker_counts = df['Ticker'].value_counts()
        df['Ticker_freq'] = df['Ticker'].map(ticker_counts)
        df['Ticker_num'] = pd.Categorical(df['Ticker']).codes
        feature_cols.extend(['Ticker_freq', 'Ticker_num'])
    
    # Volume-Price ratio
    df['Volume_Price_ratio'] = df['Volume_log'] / (df['Close'] + 1e-8)
    feature_cols.append('Volume_Price_ratio')
    
    # Clean data
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    X = df[feature_cols].copy()
    y = df['label'].copy()
    
    return X, y, feature_cols

def train_quantum_models(sample_size=1000):
    """Train multiple models and return results"""
    start_time = time.time()
    
    # Load data
    X, y, feature_cols = load_data_for_training()
    
    # Sample data
    if len(X) > sample_size:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=42)
        sample_idx, _ = next(sss.split(X, y))
        X_sample = X.iloc[sample_idx].reset_index(drop=True)
        y_sample = y.iloc[sample_idx].reset_index(drop=True)
    else:
        X_sample, y_sample = X, y
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_sample, test_size=0.2, random_state=42, stratify=y_sample
    )
    
    models = {}
    predictions = {}
    accuracies = {}
    
    # Train SVM
    svm_model = SVC(kernel='rbf', C=10.0, gamma='scale', cache_size=100, probability=True)
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    
    models['SVM'] = svm_model
    predictions['SVM'] = svm_pred
    accuracies['SVM'] = svm_accuracy
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    models['RandomForest'] = rf_model
    predictions['RandomForest'] = rf_pred
    accuracies['RandomForest'] = rf_accuracy
    
    # Train Gradient Boosting
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_accuracy = accuracy_score(y_test, gb_pred)
    
    models['GradientBoosting'] = gb_model
    predictions['GradientBoosting'] = gb_pred
    accuracies['GradientBoosting'] = gb_accuracy
    
    # Ensemble prediction
    ensemble_pred = np.round((predictions['SVM'] + predictions['RandomForest'] + predictions['GradientBoosting']) / 3)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    accuracies['Ensemble'] = ensemble_accuracy
    
    # Find best model
    best_model_name = max(accuracies, key=accuracies.get)
    best_accuracy = accuracies[best_model_name]
    best_model = models.get(best_model_name, models['SVM'])
    
    training_time = time.time() - start_time
    
    # Store results
    session_id = str(uuid.uuid4())
    trained_models[session_id] = {
        'models': models,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'best_model_name': best_model_name,
        'accuracies': accuracies,
        'training_time': training_time,
        'sample_size': len(X_sample),
        'timestamp': datetime.now().isoformat()
    }
    
    # Add to history
    training_history.append({
        'session_id': session_id,
        'timestamp': datetime.now().isoformat(),
        'sample_size': len(X_sample),
        'best_accuracy': best_accuracy,
        'best_model': best_model_name,
        'training_time': training_time
    })
    
    return {
        'session_id': session_id,
        'accuracies': {k: float(v) for k, v in accuracies.items()},
        'best_model': best_model_name,
        'best_accuracy': float(best_accuracy),
        'training_time': training_time,
        'sample_size': len(X_sample),
        'features_used': len(feature_cols),
        'memory_usage': get_memory_usage()
    }

@app.route('/')
def index():
    """Main page with modern interface"""
    return render_template('modern_index.html')

@app.route('/train', methods=['POST'])
def train_model():
    """Train quantum models endpoint"""
    try:
        data = request.get_json()
        sample_size = data.get('sample_size', 1000)
        
        # Validate sample size
        if sample_size < 100 or sample_size > 10000:
            return jsonify({'error': 'Sample size must be between 100 and 10000'}), 400
        
        results = train_quantum_models(sample_size)
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions using trained model"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if session_id not in trained_models:
            return jsonify({'error': 'Model not found. Please train a model first.'}), 404
        
        # Get input features
        features = data.get('features', {})
        model_data = trained_models[session_id]
        
        # Prepare input data
        input_array = []
        for col in model_data['feature_cols']:
            if col in features:
                input_array.append(float(features[col]))
            else:
                input_array.append(0.0)  # Default value
        
        # Scale input
        input_scaled = model_data['scaler'].transform([input_array])
        
        # Get best model
        best_model_name = model_data['best_model_name']
        if best_model_name == 'Ensemble':
            # For ensemble, use average of all models
            predictions = []
            for model_name, model in model_data['models'].items():
                pred = model.predict_proba(input_scaled)[0]
                predictions.append(pred)
            avg_pred = np.mean(predictions, axis=0)
            prediction = int(np.argmax(avg_pred))
            confidence = float(np.max(avg_pred))
        else:
            model = model_data['models'][best_model_name]
            pred_proba = model.predict_proba(input_scaled)[0]
            prediction = int(np.argmax(pred_proba))
            confidence = float(np.max(pred_proba))
        
        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'model_used': best_model_name,
            'prediction_text': 'Stock Price Will Rise' if prediction == 1 else 'Stock Price Will Fall'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def get_history():
    """Get training history"""
    return jsonify(training_history[-10:])  # Return last 10 training sessions

@app.route('/status')
def get_status():
    """Get system status"""
    return jsonify({
        'memory_usage': get_memory_usage(),
        'active_models': len(trained_models),
        'total_trainings': len(training_history)
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
