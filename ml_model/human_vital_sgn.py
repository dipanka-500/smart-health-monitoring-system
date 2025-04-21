# -*- coding: utf-8 -*-
"""
Vital Signs Analysis Pipeline

This script handles:
1. Data preprocessing and GSR simulation
2. Model training and evaluation
3. Prediction and continuous learning
"""

# Standard library imports
import os
import json
from typing import List, Dict, Tuple, Optional

# Third-party imports
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                           recall_score, f1_score,
                           classification_report, confusion_matrix)

# Constants
CONFIG = {
    "paths": {
        "raw_data": "/content/human_vital_signs_dataset_2024.csv",
        "processed_data": "human_vital_signs_dataset_with_gsr.csv",
        "model_dir": "model_results",
        "model_file": "xgb_model.json",
        "scaler_file": "scaler.joblib",
        "encoder_file": "label_encoder.joblib",
        "metrics_file": "evaluation_metrics.json"
    },
    "model_params": {
        "objective": 'multi:softmax',
        "n_estimators": 1000,
        "max_depth": 7,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": 'mlogloss'
    },
    "gsr_ranges": {
        "stressed": (10.0, 20.0),
        "relaxed": (0.5, 2.0),
        "neutral": (2.5, 9.5)
    }
}

class DataPreprocessor:
    """Handles data loading, cleaning, and feature engineering"""
    
    @staticmethod
    def simulate_gsr(row: pd.Series) -> float:
        """Simulate GSR values based on physiological markers"""
        hr = row['Heart Rate']
        temp = row['Body Temperature']
        
        if hr > 100 or temp > 37.5:
            return np.random.uniform(*CONFIG["gsr_ranges"]["stressed"])
        elif hr < 60 or temp < 36.0:
            return np.random.uniform(*CONFIG["gsr_ranges"]["relaxed"])
        return np.random.uniform(*CONFIG["gsr_ranges"]["neutral"])
    
    @classmethod
    def preprocess_data(cls, input_path: str, output_path: str) -> pd.DataFrame:
        """Load and preprocess raw data"""
        df = pd.read_csv(input_path)
        
        # Add simulated GSR data
        df['GSR (µS)'] = df.apply(cls.simulate_gsr, axis=1)
        
        # Save processed data
        df.to_csv(output_path, index=False)
        print(f"✅ Processed data saved to {output_path}")
        return df

class ModelTrainer:
    """Handles model training and evaluation"""
    
    @staticmethod
    def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and target for training"""
        # Clean data
        df = df.dropna().copy()
        df.drop(["Timestamp", "Patient ID"], axis=1, inplace=True)
        
        # Encode categorical variables
        df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
        
        # Prepare features and target
        X = df.drop("Risk Category", axis=1)
        y = df["Risk Category"]
        
        # Encode target labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        class_names = label_encoder.classes_
        
        # Normalize features
        X_scaled = StandardScaler().fit_transform(X)
        
        return X_scaled, y_encoded, class_names
    
    @classmethod
    def train_model(cls, X: np.ndarray, y: np.ndarray, num_classes: int) -> xgb.XGBClassifier:
        """Train XGBoost classifier"""
        params = CONFIG["model_params"].copy()
        params["num_class"] = num_classes
        
        model = xgb.XGBClassifier(**params)
        model.fit(X, y)
        return model
    
    @staticmethod
    def evaluate_model(model: xgb.XGBClassifier, X_test: np.ndarray, y_test: np.ndarray, 
                      class_names: List[str], save_dir: str) -> Dict[str, float]:
        """Evaluate model performance and save results"""
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted')
        }
        
        # Save evaluation results
        os.makedirs(save_dir, exist_ok=True)
        
        # Save classification report
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        with open(os.path.join(save_dir, CONFIG["paths"]["metrics_file"]), 'w') as f:
            json.dump({**metrics, "classification_report": report}, f, indent=4)
        
        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Save feature importance
        plt.figure(figsize=(12, 8))
        xgb.plot_importance(model, max_num_features=15)
        plt.title('Feature Importance')
        plt.savefig(os.path.join(save_dir, 'feature_importance.png'))
        plt.close()
        
        return metrics

class VitalSignsPredictor:
    """Handles model loading, prediction, and continuous learning"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.load_components()
    
    def load_components(self) -> None:
        """Load model and preprocessing components"""
        paths = CONFIG["paths"]
        
        if all(os.path.exists(os.path.join(paths["model_dir"], f)) 
               for f in [paths["model_file"], paths["scaler_file"], paths["encoder_file"]]):
            self.model = xgb.XGBClassifier()
            self.model.load_model(os.path.join(paths["model_dir"], paths["model_file"]))
            self.scaler = joblib.load(os.path.join(paths["model_dir"], paths["scaler_file"]))
            self.label_encoder = joblib.load(os.path.join(paths["model_dir"], paths["encoder_file"]))
            print("✅ Model and preprocessors loaded successfully")
        else:
            print("⚠️ Model not found. Please train the model first.")
    
    def predict(self, input_data: Dict) -> str:
        """Make prediction on single input sample"""
        if not all([self.model, self.scaler, self.label_encoder]):
            raise ValueError("Model components not loaded properly")
        
        # Convert input to DataFrame and ensure correct feature order
        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=self.scaler.feature_names_in_, fill_value=0)
        
        # Scale features and predict
        scaled_input = self.scaler.transform(input_df)
        pred = self.model.predict(scaled_input)
        return self.label_encoder.inverse_transform(pred)[0]
    
    def evaluate_samples(self, samples: List[Dict], true_labels: List[str]) -> None:
        """Evaluate model on test samples with optional retraining"""
        predictions = []
        
        for i, (sample, true_label) in enumerate(zip(samples, true_labels), 1):
            pred = self.predict(sample)
            predictions.append(pred)
            
            print(f"\nSample {i}:")
            print(f"Predicted: {pred}")
            print(f"Actual: {true_label}")
            
            if pred != true_label:
                print("🔁 Mismatch detected! Retraining model...")
                self.retrain_model(sample, true_label)
                pred = self.predict(sample)  # Predict again with updated model
                print(f"New prediction after retraining: {pred}")
        
        # Final evaluation
        self._print_evaluation_metrics(true_labels, predictions)
    
    def retrain_model(self, new_sample: Dict, true_label: str) -> None:
        """Retrain model with new sample"""
        # Load current training data
        df = pd.read_csv(CONFIG["paths"]["processed_data"])
        df['Gender'] = df['Gender'].replace({'Female': 0, 'Male': 1})
        
        # Add new sample
        new_sample_df = pd.DataFrame([new_sample])
        new_sample_df['Risk Category'] = true_label
        df = pd.concat([df, new_sample_df], ignore_index=True)
        
        # Prepare data
        X = df.drop(columns=['Risk Category'])
        y = df['Risk Category']
        
        # Update preprocessing objects
        y_encoded = self.label_encoder.fit_transform(y)
        X_scaled = self.scaler.fit_transform(X)
        
        # Retrain model
        num_classes = len(self.label_encoder.classes_)
        self.model = ModelTrainer.train_model(X_scaled, y_encoded, num_classes)
        
        # Save updated components
        self._save_components()
    
    def _save_components(self) -> None:
        """Save model and preprocessing components"""
        paths = CONFIG["paths"]
        os.makedirs(paths["model_dir"], exist_ok=True)
        
        self.model.save_model(os.path.join(paths["model_dir"], paths["model_file"]))
        joblib.dump(self.scaler, os.path.join(paths["model_dir"], paths["scaler_file"]))
        joblib.dump(self.label_encoder, os.path.join(paths["model_dir"], paths["encoder_file"]))
        print("✅ Model components saved successfully")
    
    def _print_evaluation_metrics(self, true_labels: List[str], predictions: List[str]) -> None:
        """Print evaluation metrics"""
        # Ensure consistent label encoding
        all_labels = list(set(true_labels + predictions))
        self.label_encoder.fit(all_labels)
        
        y_true = self.label_encoder.transform(true_labels)
        y_pred = self.label_encoder.transform(predictions)
        
        print("\n🧾 Final Evaluation:")
        print(classification_report(y_true, y_pred, target_names=self.label_encoder.classes_))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_true, y_pred),
                    annot=True, fmt='d', cmap="Blues",
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

def main():
    """Main execution pipeline"""
    # Step 1: Preprocess data and add GSR values
    print("🚀 Starting data preprocessing...")
    DataPreprocessor.preprocess_data(
        CONFIG["paths"]["raw_data"],
        CONFIG["paths"]["processed_data"]
    )
    
    # Step 2: Train and evaluate model
    print("\n🔧 Training model...")
    df = pd.read_csv(CONFIG["paths"]["processed_data"])
    X, y, class_names = ModelTrainer.prepare_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = ModelTrainer.train_model(X_train, y_train, len(class_names))
    
    # Evaluate model
    metrics = ModelTrainer.evaluate_model(
        model, X_test, y_test, class_names,
        CONFIG["paths"]["model_dir"]
    )
    
    # Save model components
    joblib.dump(model, os.path.join(CONFIG["paths"]["model_dir"], "model.joblib"))
    print(f"\n🎉 Training complete! Model accuracy: {metrics['accuracy']:.2f}")
    
    # Step 3: Demonstrate prediction and continuous learning
    print("\n🔮 Testing predictions with continuous learning...")
    predictor = VitalSignsPredictor()
    
    test_samples = [
        {
            'Patient ID': 1, 'Age': 50, 'Gender': 1, 'Heart Rate': 92, 
            'Body Temperature': 37.2, 'Oxygen Saturation': 94.5, 
            'Systolic Blood Pressure': 150, 'Diastolic Blood Pressure': 100,
            'Respiratory Rate': 22, 'Weight (kg)': 78, 'Height (m)': 1.7,
            'Derived_HRV': 0.10, 'Derived_Pulse_Pressure': 50, 
            'Derived_BMI': 26.99, 'Derived_MAP': 116.7, 'GSR (µS)': 21.0
        },
        {
            'Patient ID': 2, 'Age': 35, 'Gender': 0, 'Heart Rate': 75, 
            'Body Temperature': 36.5, 'Oxygen Saturation': 98.0, 
            'Systolic Blood Pressure': 118, 'Diastolic Blood Pressure': 76,
            'Respiratory Rate': 16, 'Weight (kg)': 60, 'Height (m)': 1.65,
            'Derived_HRV': 0.15, 'Derived_Pulse_Pressure': 42, 
            'Derived_BMI': 22.04, 'Derived_MAP': 90.0, 'GSR (µS)': 18.0
        }
    ]
    true_labels = ["Medium Risk", "Low Risk"]
    
    predictor.evaluate_samples(test_samples, true_labels)

if __name__ == "__main__":
    main()