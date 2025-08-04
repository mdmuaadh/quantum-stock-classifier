import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC  # Classical SVM for comparison
import time
import psutil
import gc
import sys

def print_status(message):
    """Print status with timestamp and flush immediately"""
    print(f"[{time.strftime('%H:%M:%S')}] {message}")
    sys.stdout.flush()

def get_memory_usage():
    """Get current memory usage percentage"""
    return psutil.virtual_memory().percent

def load_data_fast():
    """Load data with enhanced feature engineering for better accuracy"""
    print_status("Loading data with enhanced features...")
    
    # Use the smallest available dataset for maximum speed
    try:
        df = pd.read_csv('quantum_ready_sp500_subset.csv')
        print_status(f"Loaded {len(df)} samples from subset file")
    except:
        try:
            df = pd.read_csv('quantum_ready_stocks.csv')
            print_status(f"Loaded {len(df)} samples from stocks file")
        except:
            raise FileNotFoundError("No data files found")
    
    print_status(f"Available columns: {list(df.columns)}")
    
    # Enhanced feature engineering for better accuracy
    feature_cols = ['Close', 'Volume']
    
    # Log transform volume for better distribution
    df['Volume_log'] = np.log1p(df['Volume'])
    feature_cols.append('Volume_log')
    
    # Price volatility feature (if we have enough data)
    if len(df) > 1000:
        df['Close_pct_change'] = df.groupby('Ticker')['Close'].pct_change().fillna(0)
        df['Price_volatility'] = df.groupby('Ticker')['Close_pct_change'].rolling(5, min_periods=1).std().reset_index(0, drop=True).fillna(0)
        feature_cols.extend(['Close_pct_change', 'Price_volatility'])
    
    # Ticker encoding with better distribution
    if 'Ticker' in df.columns:
        # Use frequency encoding for better performance
        ticker_counts = df['Ticker'].value_counts()
        df['Ticker_freq'] = df['Ticker'].map(ticker_counts)
        df['Ticker_num'] = pd.Categorical(df['Ticker']).codes
        feature_cols.extend(['Ticker_freq', 'Ticker_num'])
    
    # Volume-Price ratio
    df['Volume_Price_ratio'] = df['Volume_log'] / (df['Close'] + 1e-8)
    feature_cols.append('Volume_Price_ratio')
    
    # Remove any infinite or NaN values
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    X = df[feature_cols].copy()
    y = df['label'].copy()
    
    print_status(f"Enhanced features ({len(feature_cols)}): {feature_cols}")
    print_status(f"Label distribution: {y.value_counts().to_dict()}")
    
    return X, y

def train_turbo_model(X, y, sample_size=1000):
    """Train ultra-fast model with immediate results"""
    print_status(f"Starting turbo training with {sample_size} samples...")
    print_status(f"Memory usage: {get_memory_usage():.1f}%")
    
    # Sample data for speed
    if len(X) > sample_size:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=42)
        sample_idx, _ = next(sss.split(X, y))
        X_sample = X.iloc[sample_idx].reset_index(drop=True)
        y_sample = y.iloc[sample_idx].reset_index(drop=True)
    else:
        X_sample, y_sample = X, y
    
    print_status(f"Using {len(X_sample)} samples")
    
    # Normalize features
    print_status("Normalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)
    
    # Split data
    print_status("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_sample, test_size=0.2, random_state=42, stratify=y_sample
    )
    
    print_status(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Train multiple optimized models for better accuracy
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    
    models = {}
    predictions = {}
    accuracies = {}
    
    # 1. Optimized SVM
    print_status("Training optimized SVM...")
    start_time = time.time()
    
    svm_model = SVC(kernel='rbf', C=10.0, gamma='scale', cache_size=100, probability=True)
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    
    models['SVM'] = svm_model
    predictions['SVM'] = svm_pred
    accuracies['SVM'] = svm_accuracy
    
    svm_time = time.time() - start_time
    print_status(f"Optimized SVM: {svm_accuracy:.4f} accuracy in {svm_time:.2f}s")
    
    # 2. Random Forest (fast and accurate)
    print_status("Training Random Forest...")
    start_time = time.time()
    
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    models['RandomForest'] = rf_model
    predictions['RandomForest'] = rf_pred
    accuracies['RandomForest'] = rf_accuracy
    
    rf_time = time.time() - start_time
    print_status(f"Random Forest: {rf_accuracy:.4f} accuracy in {rf_time:.2f}s")
    
    # 3. Gradient Boosting (often highest accuracy)
    print_status("Training Gradient Boosting...")
    start_time = time.time()
    
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_accuracy = accuracy_score(y_test, gb_pred)
    
    models['GradientBoosting'] = gb_model
    predictions['GradientBoosting'] = gb_pred
    accuracies['GradientBoosting'] = gb_accuracy
    
    gb_time = time.time() - start_time
    print_status(f"Gradient Boosting: {gb_accuracy:.4f} accuracy in {gb_time:.2f}s")
    
    # 4. Ensemble prediction (combine all models)
    print_status("Creating ensemble prediction...")
    
    # Simple voting ensemble
    ensemble_pred = np.round((predictions['SVM'] + predictions['RandomForest'] + predictions['GradientBoosting']) / 3)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    
    accuracies['Ensemble'] = ensemble_accuracy
    print_status(f"Ensemble Model: {ensemble_accuracy:.4f} accuracy")
    
    # Find best model
    best_model_name = max(accuracies, key=accuracies.get)
    best_accuracy = accuracies[best_model_name]
    best_model = models.get(best_model_name, models['SVM'])  # Fallback to SVM if ensemble
    
    # Results summary
    print_status("=" * 50)
    print_status("ENHANCED TURBO TRAINING RESULTS")
    print_status("=" * 50)
    print_status(f"Dataset size: {len(X_sample)} samples")
    print_status(f"Features used: {X_sample.shape[1]}")
    
    # Show all model results
    for model_name, accuracy in accuracies.items():
        print_status(f"{model_name}: {accuracy*100:.2f}% accuracy")
    
    print_status(f"Memory usage: {get_memory_usage():.1f}%")
    
    # Detailed report for best model
    if best_model_name == 'Ensemble':
        best_pred = ensemble_pred
    else:
        best_pred = predictions[best_model_name]
    
    print_status(f"\nBest model: {best_model_name} ({best_accuracy*100:.2f}% accuracy)")
    print_status("Classification Report:")
    print(classification_report(y_test, best_pred))
    
    return best_model, best_accuracy, accuracies

def main():
    """Main turbo training function"""
    print_status("=== TURBO QUANTUM TRAINER ===")
    print_status("Ultra-fast training with immediate results")
    print_status("Target: 30-60 seconds, good accuracy, minimal resources")
    print_status("=" * 50)
    
    total_start = time.time()
    
    try:
        # Load data
        X, y = load_data_fast()
        
        # Try different sample sizes for speed vs accuracy trade-off
        sample_sizes = [500, 1000, 2000]
        results = []
        
        for size in sample_sizes:
            if len(X) >= size:
                print_status(f"\n--- Testing with {size} samples ---")
                model, accuracy, all_accuracies = train_turbo_model(X, y, size)
                results.append((size, accuracy, model, all_accuracies))
                
                # If we get good accuracy quickly, we can stop
                if accuracy > 0.80:
                    print_status(f"Excellent accuracy achieved! Stopping early.")
                    break
        
        # Final summary
        total_time = time.time() - total_start
        print_status("\n" + "=" * 50)
        print_status("TURBO TRAINING COMPLETE!")
        print_status(f"Total time: {total_time:.1f} seconds")
        
        if results:
            best_size, best_acc, best_model, best_all_acc = max(results, key=lambda x: x[1])
            print_status(f"Best result: {best_acc*100:.2f}% with {best_size} samples")
            print_status(f"All model accuracies: {', '.join([f'{k}: {v*100:.1f}%' for k, v in best_all_acc.items()])}")
        
        print_status(f"Final memory usage: {get_memory_usage():.1f}%")
        print_status("=" * 50)
        
        return results
        
    except Exception as e:
        print_status(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print_status(f"Starting with {get_memory_usage():.1f}% memory usage")
    results = main()
    
    if results:
        print_status("SUCCESS! Fast training completed.")
    else:
        print_status("Training failed - check errors above.")
