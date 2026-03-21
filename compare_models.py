#!/usr/bin/env python
"""
Comprehensive comparison between Quantum SVM and Classical ML models
for Tomato Leaf Disease Classification
"""

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import Paths, FeatureConfig, ImageConfig, TrainConfig
from src.dataset import discover_dataset, load_image_record
from src.features import extract_features
from src.preprocess import preprocess_for_features


class ModelComparison:
    """Compare QSVM with classical ML models"""
    
    def __init__(self):
        self.paths = Paths()
        self.img_cfg = ImageConfig()
        self.feat_cfg = FeatureConfig()
        self.tr_cfg = TrainConfig()
        self.results = {}
        self.X_test_p = None
        self.y_test = None
        self.le = None
        
    def load_data(self):
        """Load and prepare dataset"""
        print("=" * 80)
        print("TOMATO LEAF DISEASE CLASSIFICATION - MODEL COMPARISON")
        print("=" * 80)
        print("\n[1/5] Loading dataset...")
        
        records = discover_dataset(self.paths)
        print(f"[OK] Found {len(records)} images")
        
        # Cap samples per class
        if self.tr_cfg.max_samples_per_class and self.tr_cfg.max_samples_per_class > 0:
            from collections import defaultdict
            by_label = defaultdict(list)
            for r in records:
                by_label[r.label].append(r)
            capped = []
            rng = np.random.default_rng(self.tr_cfg.random_state)
            for label, recs in sorted(by_label.items(), key=lambda x: x[0]):
                if len(recs) > self.tr_cfg.max_samples_per_class:
                    idx = rng.choice(len(recs), size=self.tr_cfg.max_samples_per_class, replace=False)
                    capped.extend([recs[i] for i in idx])
                else:
                    capped.extend(recs)
            records = capped
        
        # Extract features
        X_list = []
        y_list = []
        for i, rec in enumerate(records):
            rgb = load_image_record(rec)
            pre = preprocess_for_features(rgb, img_cfg=self.img_cfg)
            feats = extract_features(pre, cfg=self.feat_cfg)
            X_list.append(feats)
            y_list.append(rec.label)
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(records)} images", end='\r')
        
        X = np.stack(X_list, axis=0)
        y_str = np.array(y_list)
        
        # Encode labels
        self.le = LabelEncoder()
        y = self.le.fit_transform(y_str)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.tr_cfg.test_size, random_state=self.tr_cfg.random_state, stratify=y
        )
        
        # Scaling
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # PCA
        pca = PCA(n_components=self.tr_cfg.pca_components, random_state=self.tr_cfg.random_state)
        X_train_p = pca.fit_transform(X_train_s)
        self.X_test_p = pca.transform(X_test_s)
        
        self.y_test = y_test
        
        print(f"\n[OK] Dataset loaded and prepared")
        print(f"  Training samples: {X_train_p.shape[0]}")
        print(f"  Test samples: {self.X_test_p.shape[0]}")
        print(f"  Features (after PCA): {X_train_p.shape[1]}")
        print(f"  Classes: {len(self.le.classes_)}")
        print(f"  Classes: {', '.join(self.le.classes_)}")
        
        return X_train_p, y_train
    
    def test_qsvm(self, X_train, y_train):
        """Test trained QSVM model"""
        print("\n[2/5] Testing Quantum SVM (QSVM)...")
        
        try:
            bundle = joblib.load(self.paths.model_path)
            qsvc = bundle["qsvc"]
            
            start_time = time.time()
            y_pred = qsvc.predict(self.X_test_p)
            inference_time = time.time() - start_time
            
            self.results['QSVM'] = {
                'predictions': y_pred,
                'inference_time': inference_time,
                'model_type': 'Quantum SVM (Qiskit)',
                'training_time': 'Pre-trained (not retrained)',
            }
            
            print(f"[OK] QSVM tested successfully")
            print(f"  Inference time: {inference_time:.4f} seconds")
            print(f"  Predictions made: {len(y_pred)}")
            
        except Exception as e:
            print(f"[ERROR] Error testing QSVM: {e}")
    
    def test_classical_svm(self, X_train, y_train):
        """Train and test classical SVM"""
        print("\n[3/5] Training Classical SVM (RBF kernel)...")
        
        try:
            start_time = time.time()
            svm = SVC(kernel='rbf', C=100, gamma='scale', probability=True)
            svm.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            start_time = time.time()
            y_pred = svm.predict(self.X_test_p)
            inference_time = time.time() - start_time
            
            self.results['Classical SVM (RBF)'] = {
                'predictions': y_pred,
                'training_time': training_time,
                'inference_time': inference_time,
                'model_type': 'Classical SVM - RBF Kernel',
            }
            
            print(f"[OK] Classical SVM trained and tested")
            print(f"  Training time: {training_time:.4f} seconds")
            print(f"  Inference time: {inference_time:.4f} seconds")
            
        except Exception as e:
            print(f"[ERROR] Error with Classical SVM: {e}")
    
    def test_svm_linear(self, X_train, y_train):
        """Train and test SVM with linear kernel"""
        print("\n[3b/5] Training Classical SVM (Linear kernel)...")
        
        try:
            start_time = time.time()
            svm = SVC(kernel='linear', C=100, probability=True)
            svm.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            start_time = time.time()
            y_pred = svm.predict(self.X_test_p)
            inference_time = time.time() - start_time
            
            self.results['Classical SVM (Linear)'] = {
                'predictions': y_pred,
                'training_time': training_time,
                'inference_time': inference_time,
                'model_type': 'Classical SVM - Linear Kernel',
            }
            
            print(f"[OK] Classical SVM (Linear) trained and tested")
            print(f"  Training time: {training_time:.4f} seconds")
            print(f"  Inference time: {inference_time:.4f} seconds")
            
        except Exception as e:
            print(f"[ERROR] Error with Classical SVM (Linear): {e}")
    
    def test_random_forest(self, X_train, y_train):
        """Train and test Random Forest"""
        print("\n[3c/5] Training Random Forest Classifier...")
        
        try:
            start_time = time.time()
            rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
            rf.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            start_time = time.time()
            y_pred = rf.predict(self.X_test_p)
            inference_time = time.time() - start_time
            
            self.results['Random Forest'] = {
                'predictions': y_pred,
                'training_time': training_time,
                'inference_time': inference_time,
                'model_type': 'Random Forest (100 trees)',
            }
            
            print(f"[OK] Random Forest trained and tested")
            print(f"  Training time: {training_time:.4f} seconds")
            print(f"  Inference time: {inference_time:.4f} seconds")
            
        except Exception as e:
            print(f"[ERROR] Error with Random Forest: {e}")
    
    def calculate_metrics(self):
        """Calculate and display metrics for all models"""
        print("\n[4/5] Calculating evaluation metrics...")
        
        metrics_summary = {}
        
        for model_name, result in self.results.items():
            y_pred = result['predictions']
            
            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            
            metrics_summary[model_name] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
                'inference_time': result.get('inference_time', 0),
                'training_time': result.get('training_time', 'N/A'),
            }
        
        # Display metrics table
        print("\n" + "=" * 120)
        print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Inference Time':<15}")
        print("=" * 120)
        
        for model_name, metrics in metrics_summary.items():
            acc = f"{metrics['accuracy']:.4f}"
            prec = f"{metrics['precision']:.4f}"
            rec = f"{metrics['recall']:.4f}"
            f1 = f"{metrics['f1_score']:.4f}"
            inf_time = f"{metrics['inference_time']:.4f}s"
            print(f"{model_name:<25} {acc:<12} {prec:<12} {rec:<12} {f1:<12} {inf_time:<15}")
        
        print("=" * 120)
        
        # Find best model
        best_model = max(metrics_summary.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n[BEST] Best Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")
        
        return metrics_summary
    
    def print_detailed_report(self, metrics_summary):
        """Print detailed classification report"""
        print("\n[5/5] Detailed Classification Reports:")
        print("=" * 120)
        
        for model_name, result in self.results.items():
            y_pred = result['predictions']
            
            print(f"\n[METRICS] {model_name}")
            print("-" * 80)
            print(classification_report(self.y_test, y_pred, target_names=self.le.classes_, zero_division=0))
    
    def save_comparison_report(self, metrics_summary):
        """Save comparison report as JSON"""
        report = {
            'timestamp': str(Path(__file__).parent),
            'model_comparison': {}
        }
        
        for model_name, metrics in metrics_summary.items():
            report['model_comparison'][model_name] = {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1_score': float(metrics['f1_score']),
                'inference_time_seconds': float(metrics['inference_time']),
            }
        
        output_file = self.paths.artifacts_dir / 'model_comparison.json'
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n[OK] Comparison report saved to: {output_file}")
    
    def run(self):
        """Run complete comparison"""
        X_train, y_train = self.load_data()
        self.test_qsvm(X_train, y_train)
        self.test_classical_svm(X_train, y_train)
        self.test_svm_linear(X_train, y_train)
        self.test_random_forest(X_train, y_train)
        metrics_summary = self.calculate_metrics()
        self.print_detailed_report(metrics_summary)
        self.save_comparison_report(metrics_summary)
        
        print("\n" + "=" * 80)
        print("[OK] Model comparison completed!")
        print("=" * 80)


if __name__ == '__main__':
    comparison = ModelComparison()
    comparison.run()
