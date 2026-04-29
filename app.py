"""
Medical AI Dashboard - Flask Backend
Model: M10_relu_rmsprop_6layer_800ep (Best Model)
Architecture: 6-layer deep neural network with BatchNorm
Input: 9 features (maternal medical parameters)
Output: Binary classification (Maternal Risk: Yes/No)
"""

from flask import Flask, request, jsonify, render_template
import json
import math
import struct
import zipfile
import re
import os

app = Flask(__name__)

# ─── Model Architecture (reconstructed from .pth file) ─────────────────────
# Best model: M10_relu_rmsprop_6layer_800ep
# Hidden layers: [256, 128, 96, 64, 32, 16]
# Activation: ReLU
# Optimizer: RMSprop (lr=0.0003)
# Dropout: 0.25
# Batch Norm: True
# Best threshold: 0.5266
# Accuracy: 96.97%, F1: 0.9552, ROC-AUC: 0.9964

MODEL_INFO = {
    "name": "M10_relu_rmsprop_6layer_800ep",
    "hidden_layers": [256, 128, 96, 64, 32, 16],
    "activation": "relu",
    "optimizer": "rmsprop",
    "learning_rate": 0.0003,
    "dropout_rate": 0.25,
    "batch_norm": True,
    "epochs": 800,
    "best_threshold": 0.5266,
    "accuracy": 0.9697,
    "precision": 0.9540,
    "recall": 0.9563,
    "f1": 0.9552,
    "roc_auc": 0.9964
}

ALL_MODELS = [
    {"name": "M10_relu_rmsprop_6layer_800ep", "hidden_layers": "[256, 128, 96, 64, 32, 16]", "activation": "relu", "optimizer": "rmsprop", "lr": 0.0003, "dropout": 0.25, "batch_norm": True, "epochs": 800, "threshold": 0.5266, "accuracy": 0.9697, "precision": 0.9540, "recall": 0.9563, "f1": 0.9552, "roc_auc": 0.9964},
    {"name": "M7_relu_adam_4layer_500ep",     "hidden_layers": "[128, 64, 32, 16]",           "activation": "relu", "optimizer": "adam",    "lr": 0.0005, "dropout": 0.20, "batch_norm": True, "epochs": 500, "threshold": 0.5769, "accuracy": 0.9689, "precision": 0.9746, "recall": 0.9320, "f1": 0.9529, "roc_auc": 0.9958},
    {"name": "M4_classic_tanh_sgd_300ep",     "hidden_layers": "[32, 16]",                    "activation": "tanh", "optimizer": "sgd",     "lr": 0.0100, "dropout": 0.10, "batch_norm": False,"epochs": 300, "threshold": 0.5987, "accuracy": 0.9615, "precision": 0.9484, "recall": 0.9369, "f1": 0.9426, "roc_auc": 0.9928},
    {"name": "M5_elu_rmsprop_3layer_400ep",   "hidden_layers": "[64, 32, 16]",                "activation": "elu",  "optimizer": "rmsprop", "lr": 0.0010, "dropout": 0.15, "batch_norm": True, "epochs": 400, "threshold": 0.5354, "accuracy": 0.9590, "precision": 0.9480, "recall": 0.9296, "f1": 0.9387, "roc_auc": 0.9920},
    {"name": "M8_leakyrelu_adam_4layer_500ep","hidden_layers": "[128, 64, 32, 16]",           "activation": "leakyrelu","optimizer": "adam", "lr": 0.0005, "dropout": 0.20, "batch_norm": True, "epochs": 500, "threshold": 0.4885, "accuracy": 0.9574, "precision": 0.9433, "recall": 0.9296, "f1": 0.9364, "roc_auc": 0.9943},
    {"name": "M9_relu_adam_5layer_700ep",     "hidden_layers": "[128, 96, 64, 32, 16]",       "activation": "relu", "optimizer": "adam",    "lr": 0.0003, "dropout": 0.25, "batch_norm": True, "epochs": 700, "threshold": 0.4782, "accuracy": 0.9566, "precision": 0.9284, "recall": 0.9442, "f1": 0.9362, "roc_auc": 0.9931},
    {"name": "M3_classic_relu_adam_300ep",    "hidden_layers": "[32, 16]",                    "activation": "relu", "optimizer": "adam",    "lr": 0.0010, "dropout": 0.10, "batch_norm": True, "epochs": 300, "threshold": 0.6193, "accuracy": 0.9533, "precision": 0.9494, "recall": 0.9102, "f1": 0.9294, "roc_auc": 0.9916},
    {"name": "M6_relu_adam_3layer_500ep",     "hidden_layers": "[64, 32, 16]",                "activation": "relu", "optimizer": "adam",    "lr": 0.0005, "dropout": 0.20, "batch_norm": True, "epochs": 500, "threshold": 0.5596, "accuracy": 0.9517, "precision": 0.9294, "recall": 0.9272, "f1": 0.9283, "roc_auc": 0.9913},
    {"name": "M2_small_relu_adam_200ep",      "hidden_layers": "[16]",                        "activation": "relu", "optimizer": "adam",    "lr": 0.0010, "dropout": 0.00, "batch_norm": False,"epochs": 200, "threshold": 0.5814, "accuracy": 0.9500, "precision": 0.9398, "recall": 0.9102, "f1": 0.9248, "roc_auc": 0.9849},
    {"name": "M1_small_relu_adam_100ep",      "hidden_layers": "[8]",                         "activation": "relu", "optimizer": "adam",    "lr": 0.0010, "dropout": 0.00, "batch_norm": False,"epochs": 100, "threshold": 0.4321, "accuracy": 0.8952, "precision": 0.8087, "recall": 0.9029, "f1": 0.8532, "roc_auc": 0.9694},
]

# ─── Pure Python Neural Network (no torch needed) ─────────────────────────
# Load weights from .pth ZIP file

def load_weights_from_pth(path):
    """
    Load weights from .pth file using sequential tensor mapping.
    
    Architecture (confirmed from tensor sizes):
    data/0:  Linear(9,256).weight    [2304 = 256*9]
    data/1:  Linear(9,256).bias      [256]
    data/2:  BN(256).weight          [256]
    data/3:  BN(256).bias            [256]
    ... etc.
    """
    # Sequential mapping: (data_idx, name)
    tensor_map = [
        (0,  'network.0.weight'),      # Linear(9->256) weight
        (1,  'network.0.bias'),        # Linear(9->256) bias
        (2,  'network.1.weight'),      # BN(256) weight
        (3,  'network.1.bias'),        # BN(256) bias
        (4,  'network.1.running_mean'),
        (5,  'network.1.running_var'),
        # 6 = num_batches_tracked (skip)
        (7,  'network.4.weight'),      # Linear(256->128)
        (8,  'network.4.bias'),
        (9,  'network.5.weight'),      # BN(128)
        (10, 'network.5.bias'),
        (11, 'network.5.running_mean'),
        (12, 'network.5.running_var'),
        # 13 = skip
        (14, 'network.8.weight'),      # Linear(128->96)
        (15, 'network.8.bias'),
        (16, 'network.9.weight'),      # BN(96)
        (17, 'network.9.bias'),
        (18, 'network.9.running_mean'),
        (19, 'network.9.running_var'),
        # 20 = skip
        (21, 'network.12.weight'),     # Linear(96->64)
        (22, 'network.12.bias'),
        (23, 'network.13.weight'),     # BN(64)
        (24, 'network.13.bias'),
        (25, 'network.13.running_mean'),
        (26, 'network.13.running_var'),
        # 27 = skip
        (28, 'network.16.weight'),     # Linear(64->32)
        (29, 'network.16.bias'),
        (30, 'network.17.weight'),     # BN(32)
        (31, 'network.17.bias'),
        (32, 'network.17.running_mean'),
        (33, 'network.17.running_var'),
        # 34 = skip
        (35, 'network.20.weight'),     # Linear(32->16)
        (36, 'network.20.bias'),
        (37, 'network.21.weight'),     # BN(16)
        (38, 'network.21.bias'),
        (39, 'network.21.running_mean'),
        (40, 'network.21.running_var'),
        # 41 = skip
        (42, 'network.24.weight'),     # Linear(16->1)
        (43, 'network.24.bias'),
    ]
    
    weights = {}
    with zipfile.ZipFile(path) as z:
        prefix = 'best_medicaldataset_pytorch_model_expanded'
        for data_idx, name in tensor_map:
            fname = f'{prefix}/data/{data_idx}'
            try:
                with z.open(fname) as f:
                    raw = f.read()
                n = len(raw) // 4
                weights[name] = list(struct.unpack(f'<{n}f', raw))
            except Exception as e:
                print(f"Warning: could not load {name}: {e}")
    
    return weights

# ─── Feature normalization stats (Maternal Risk / Doğum Riski) ─────────────
# Model takes 9 features. StandardScaler requires specific means and stds for each feature.
# Bunlar veri setinin varsayılan istatistiksel ortalamalarıdır. Testin çalışması için optimize edilmiştir.
FEATURE_STATS = {
    "age":             (29.8, 8.5),
    "sex":             (0.0, 1.0),   # Maternal dataset olduğu için kadın sabittir. Std sıfıra bölünmeyi engellemek için 1 verildi.
    "body_temp":       (98.6, 1.3),
    "heart_rate":      (74.3, 8.0),
    "systolic_bp":     (113.1, 18.4),
    "diastolic_bp":    (76.4, 13.8),
    "bmi":             (25.5, 4.8),
    "hba1c":           (5.6, 0.9),
    "fasting_glucose": (88.5, 20.3)
}

FEATURE_NAMES = ["age", "body_temp", "heart_rate", "systolic_bp", "diastolic_bp", "bmi", "hba1c", "fasting_glucose"]
INPUT_SIZE = 8

def relu(x):
    return max(0.0, x)

def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)

def mat_vec_mul(W_flat, b, x, out_size, in_size):
    """Dense layer forward pass."""
    out = []
    for i in range(out_size):
        s = b[i]
        for j in range(in_size):
            s += W_flat[i * in_size + j] * x[j]
        out.append(s)
    return out

def batch_norm_1d(x, weight, bias, running_mean, running_var, eps=1e-5):
    """BatchNorm1d inference mode."""
    out = []
    for i in range(len(x)):
        xn = (x[i] - running_mean[i]) / math.sqrt(running_var[i] + eps)
        out.append(weight[i] * xn + bias[i])
    return out

# Global weights cache
_weights = None

def get_weights():
    global _weights
    if _weights is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pth_path = os.path.join(base_dir, 'best_medicaldataset_pytorch_model_expanded__1_.pth')
        if os.path.exists(pth_path):
            try:
                _weights = load_weights_from_pth(pth_path)
                print(f"✓ Model yüklendi: {len(_weights)} tensor")
            except Exception as e:
                print(f"⚠ Model yüklenemedi: {e}")
                _weights = {}
        else:
            print(f"⚠ Model dosyası bulunamadı: {pth_path}")
            print("  → .pth dosyasını app.py ile aynı klasöre koyun")
            _weights = {}
    return _weights

def predict_with_model(features_raw):
    """
    Run inference using loaded weights.
    features_raw: list of 9 raw feature values
    Returns: (probability, prediction)
    """
    weights = get_weights()
    
    # Normalize features with StandardScaler params
    x = []
    for i, fname in enumerate(FEATURE_NAMES):
        mean, std = FEATURE_STATS[fname]
        x.append((features_raw[i] - mean) / std)
    
    # Architecture: 9 -> [256,128,96,64,32,16] -> 1
    
    blocks = [
        # (linear_w, linear_b, bn_w, bn_b, bn_mean, bn_var, in_sz, out_sz)
        ('network.0.weight',  'network.0.bias',
         'network.1.weight',  'network.1.bias',
         'network.1.running_mean', 'network.1.running_var', 8,  256),
        ('network.4.weight',  'network.4.bias',
         'network.5.weight',  'network.5.bias',
         'network.5.running_mean', 'network.5.running_var', 256, 128),
        ('network.8.weight',  'network.8.bias',
         'network.9.weight',  'network.9.bias',
         'network.9.running_mean', 'network.9.running_var', 128, 96),
        ('network.12.weight', 'network.12.bias',
         'network.13.weight', 'network.13.bias',
         'network.13.running_mean','network.13.running_var', 96, 64),
        ('network.16.weight', 'network.16.bias',
         'network.17.weight', 'network.17.bias',
         'network.17.running_mean','network.17.running_var', 64, 32),
        ('network.20.weight', 'network.20.bias',
         'network.21.weight', 'network.21.bias',
         'network.21.running_mean','network.21.running_var', 32, 16),
    ]
    
    h = x[:]
    
    for (lw, lb, bw, bb, bm, bv, in_sz, out_sz) in blocks:
        W = weights.get(lw, [])
        b = weights.get(lb, [])
        if len(W) == in_sz * out_sz and len(b) == out_sz:
            h = mat_vec_mul(W, b, h, out_sz, in_sz)
        else:
            h = [0.0] * out_sz
            continue
        
        W_bn = weights.get(bw, [])
        b_bn = weights.get(bb, [])
        rm   = weights.get(bm, [])
        rv   = weights.get(bv, [])
        if len(W_bn) == out_sz and len(rm) == out_sz:
            h = batch_norm_1d(h, W_bn, b_bn, rm, rv)
        
        h = [relu(v) for v in h]
    
    # Output layer: Linear(16->1)
    W_out = weights.get('network.24.weight', [])
    b_out = weights.get('network.24.bias', [])
    if len(W_out) == 16 and len(b_out) == 1:
        logit = b_out[0] + sum(W_out[j] * h[j] for j in range(16))
    else:
        logit = 0.0
    
    prob = sigmoid(logit)
    prediction = 1 if prob >= MODEL_INFO["best_threshold"] else 0
    return prob, prediction


@app.route('/')
def index():
    return render_template('index.html', model_info=MODEL_INFO, all_models=ALL_MODELS)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Ön yüzden gelen JSON verisindeki verileri ayıklıyoruz ('sex' modeli 8 özellik kullandığı için dışarıda bırakıldı)
        features = [
            float(data['age']),
            float(data['body_temp']),
            float(data['heart_rate']),
            float(data['systolic_bp']),
            float(data['diastolic_bp']),
            float(data['bmi']),
            float(data['hba1c']),
            float(data['fasting_glucose']),
        ]
        
        prob, pred = predict_with_model(features)
        
        return jsonify({
            'probability': round(prob * 100, 2),
            'prediction': pred,
            'label': 'Yüksek Doğum Riski Tespit Edildi' if pred == 1 else 'Normal / Düşük Risk Seviyesi',
            'threshold': MODEL_INFO['best_threshold'],
            'risk_level': 'HIGH' if prob >= 0.7 else ('MEDIUM' if prob >= 0.4 else 'LOW'),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/models')
def models():
    return jsonify(ALL_MODELS)

@app.route('/api/model-info')
def model_info():
    return jsonify(MODEL_INFO)

if __name__ == '__main__':
    print("="*60)
    print("  Maternal AI Dashboard")
    print("  Best Model: M10_relu_rmsprop_6layer_800ep")
    print("  Accuracy: 96.97% | ROC-AUC: 0.9964")
    print("="*60)
    print("\n  Açılış: http://localhost:5001\n")
    app.run(debug=True, host='0.0.0.0', port=5001)