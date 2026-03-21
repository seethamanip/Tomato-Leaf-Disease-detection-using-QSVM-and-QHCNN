import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, f1_score

from src.config import Paths
from src.dataset import discover_dataset, load_image_record
from src.infer import predict, load_bundle
from src.infer_qhcnn import predict_qhcnn

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def main():
    paths = Paths()
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading dataset for ensemble optimization...")
    records = discover_dataset(paths)
    
    # We will use a small subset to find the best weight quickly.
    # Take 10 samples per class max.
    from collections import defaultdict
    by_label = defaultdict(list)
    for r in records:
        by_label[r.label].append(r)
        
    rng = np.random.default_rng(42)
    subset = []
    cap = 15
    for lbl, recs in by_label.items():
        if len(recs) > cap:
            chosen = rng.choice(recs, size=cap, replace=False)
            subset.extend(chosen)
        else:
            subset.extend(recs)
            
    print(f"Selected {len(subset)} images for optimizing ensemble weights.")
    
    y_true_labels = []
    qsvm_probs_list = []
    qhcnn_probs_list = []
    
    classes_order = sorted(list(by_label.keys()))
    
    for i, rec in enumerate(subset):
        rgb = load_image_record(rec)
        try:
            # QSVM prediction
            res_qsvm, _ = predict(rgb)
            # QHCNN prediction
            res_qhcnn = predict_qhcnn(rgb)
            
            # Extract probability vectors aligned to classes_order
            q_vec = np.array([res_qsvm.per_class_similarity.get(c, 0.0) for c in classes_order])
            # Normalize qsvm similarities to approx probabilities
            q_vec = softmax(q_vec)
            
            h_vec = np.array([res_qhcnn.per_class_probability.get(c, 0.0) for c in classes_order])
            # Qhcnn outputs are already probabilities normally, but ensure it sum to 1
            if h_vec.sum() > 0:
                h_vec = h_vec / h_vec.sum()
                
            qsvm_probs_list.append(q_vec)
            qhcnn_probs_list.append(h_vec)
            y_true_labels.append(classes_order.index(rec.label))
        except Exception as e:
            print(f"Error processing {rec.path}: {e}")
            
        if (i+1) % 10 == 0:
            print(f" Processed {i+1}/{len(subset)}")
            
    Q = np.array(qsvm_probs_list)
    H = np.array(qhcnn_probs_list)
    Y = np.array(y_true_labels)
    
    # Objective function to minimize (negative F1 score)
    def objective(w):
        # w is weight for QSVM, (1-w) is weight for QHCNN
        ensemble_probs = w * Q + (1 - w) * H
        preds = np.argmax(ensemble_probs, axis=1)
        return -f1_score(Y, preds, average="weighted")
        
    print("Optimizing ensemble weight using SciPy minimize_scalar...")
    res = minimize_scalar(objective, bounds=(0, 1), method='bounded')
    
    best_w = float(res.x)
    best_f1 = -float(res.fun)
    
    ensemble_probs = best_w * Q + (1 - best_w) * H
    final_preds = np.argmax(ensemble_probs, axis=1)
    best_acc = accuracy_score(Y, final_preds)
    
    print(f"Optimal QSVM Weight: {best_w:.4f}")
    print(f"Optimal QHCNN Weight: {1-best_w:.4f}")
    print(f"Validation F1-Score: {best_f1:.4f}")
    print(f"Validation Accuracy: {best_acc:.4f}")
    
    # Compute base metrics to compare
    qsvm_preds = np.argmax(Q, axis=1)
    qhcnn_preds = np.argmax(H, axis=1)
    print(f"Base QSVM Acc: {accuracy_score(Y, qsvm_preds):.4f}")
    print(f"Base QHCNN Acc: {accuracy_score(Y, qhcnn_preds):.4f}")
    
    metrics = {
        "qsvm_weight": best_w,
        "qhcnn_weight": 1.0 - best_w,
        "validation_f1": best_f1,
        "validation_accuracy": float(best_acc),
        "timestamp": time.time()
    }
    
    out_path = paths.artifacts_dir / "ensemble_weights.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Saved optimized ensemble weights to {out_path}")

if __name__ == "__main__":
    main()
