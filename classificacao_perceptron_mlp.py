"""
=============================================================================
AV3 - Inteligência Artificial Computacional - UNIFOR
Classificação: Perceptron Simples e MLP (implementação manual)
Dataset: SBA-Loans-Case-Data-Set (UCI ID: 43539)
Alunos: Paulo Henrico Rabelo (2312652) | Levi Tabosa (2224207)
=============================================================================

COMO USAR:
  1. Baixe o dataset SBA-Loans do UCI (ID: 43539) e salve como 'sba_loans.csv'
     no mesmo diretório deste script, OU ajuste SBA_CSV_PATH abaixo.
  2. Execute: python classificacao_perceptron_mlp.py
=============================================================================
"""

import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIGURAÇÃO DO CAMINHO DO DATASET ──────────────────────────────────────
SBA_CSV_PATH = "sba_loans.csv"   # <-- ajuste se necessário

# ─── SEMENTE GLOBAL ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)


# =============================================================================
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO
# =============================================================================

def load_and_preprocess(path: str):
    """
    Lê o CSV do SBA-Loans, realiza limpeza e encoding mínimo,
    e retorna X (numpy float64) e y (numpy int: 0 ou 1).

    Coluna alvo: 'MIS_Status'  (P I F → 0=pago, CHGOFF → 1=calote)
    Colunas com $ e vírgulas são convertidas para float.
    Variáveis categóricas recebem label-encoding simples.
    """
    df = pd.read_csv(path, low_memory=False)

    print(f"Dataset carregado: {df.shape[0]} linhas × {df.shape[1]} colunas")

    # --- Coluna-alvo ---
    target_col = "MIS_Status"
    if target_col not in df.columns:
        # Tenta encontrar coluna similar
        candidates = [c for c in df.columns if "mis" in c.lower() or "status" in c.lower()]
        if candidates:
            target_col = candidates[0]
            print(f"  Coluna alvo detectada: '{target_col}'")
        else:
            raise ValueError("Coluna 'MIS_Status' não encontrada. Verifique o CSV.")

    df = df.dropna(subset=[target_col])
    df[target_col] = df[target_col].str.strip()
    df = df[df[target_col].isin(["P I F", "CHGOFF"])]
    y_raw = (df[target_col] == "CHGOFF").astype(int).values
    df = df.drop(columns=[target_col])

    # --- Colunas desnecessárias ---
    drop_cols = ["LoanNr_ChkDgt", "Name", "City", "State", "Zip", "Bank",
                 "BankState", "NAICS", "ApprovalDate", "ApprovalFY",
                 "DisbursementDate", "BalanceGross", "ChgOffDate",
                 "ChgOffPrinGr", "SBA_Appv"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # --- Limpeza de valores monetários ($, vírgulas) ---
    def clean_money(series):
        return (series.astype(str)
                .str.replace(r"[\$,]", "", regex=True)
                .str.strip()
                .replace("nan", np.nan)
                .astype(float))

    money_cols = [c for c in df.columns
                  if df[c].dtype == object and
                  df[c].astype(str).str.contains(r"\$", na=False).any()]
    for c in money_cols:
        df[c] = clean_money(df[c])

    # --- Encoding de categóricas restantes ---
    for c in df.select_dtypes(include="object").columns:
        df[c] = pd.Categorical(df[c]).codes.astype(float)
        df[c] = df[c].replace(-1, np.nan)

    # --- Imputação pela mediana e remoção de NaN extremos ---
    df = df.fillna(df.median(numeric_only=True))
    df = df.select_dtypes(include=[np.number])

    X = df.values.astype(np.float64)
    print(f"Após pré-processamento: {X.shape[0]} amostras × {X.shape[1]} features")
    print(f"Distribuição alvo  →  Pago (0): {(y_raw==0).sum()}  |  Calote (1): {(y_raw==1).sum()}\n")
    return X, y_raw


def normalize(X_train, X_test):
    """Min-Max normalização [0, 1] calculada apenas no treino."""
    mn = X_train.min(axis=0)
    mx = X_train.max(axis=0)
    rng = np.where(mx - mn == 0, 1, mx - mn)
    return (X_train - mn) / rng, (X_test - mn) / rng


# =============================================================================
# 2. MÉTRICAS (implementação manual)
# =============================================================================

def confusion_matrix_manual(y_true, y_pred):
    """Retorna TP, FP, TN, FN."""
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    return TP, FP, TN, FN


def compute_metrics(y_true, y_pred):
    TP, FP, TN, FN = confusion_matrix_manual(y_true, y_pred)
    acc       = (TP + TN) / (TP + FP + TN + FN + 1e-12)
    precision = TP / (TP + FP + 1e-12)
    recall    = TP / (TP + FN + 1e-12)   # Sensibilidade
    specificity = TN / (TN + FP + 1e-12)
    f1        = 2 * precision * recall / (precision + recall + 1e-12)
    return {
        "Acurácia":      acc,
        "Precisão":      precision,
        "Recall":        recall,
        "Especificidade": specificity,
        "F1-Score":      f1,
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
    }


def print_confusion_matrix(TP, FP, TN, FN):
    print("  Matriz de Confusão:")
    print(f"                 Previsto 0   Previsto 1")
    print(f"  Real 0 (Pago)     {TN:>6}       {FP:>6}")
    print(f"  Real 1 (Calote)   {FN:>6}       {TP:>6}")


# =============================================================================
# 3. VALIDAÇÃO CRUZADA ESTRATIFICADA (5-Fold, manual)
# =============================================================================

def stratified_kfold_indices(y, k=5, seed=SEED):
    """Retorna lista de (train_idx, test_idx) estratificados."""
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    fold_indices = [[] for _ in range(k)]
    for cls in classes:
        idx = np.where(y == cls)[0]
        idx = rng.permutation(idx)
        splits = np.array_split(idx, k)
        for i, s in enumerate(splits):
            fold_indices[i].extend(s.tolist())
    folds = []
    for i in range(k):
        test_idx  = np.array(fold_indices[i])
        train_idx = np.array([j for fold in fold_indices[:i] + fold_indices[i+1:] for j in fold])
        folds.append((train_idx, test_idx))
    return folds


def run_cv(model_class, X, y, model_kwargs, k=5):
    """
    Executa k-Fold CV, retorna dicionário com médias e desvios das métricas,
    além de tempo de treino e teste.
    """
    folds = stratified_kfold_indices(y, k=k)
    all_metrics = []
    train_times, test_times = [], []

    for fold_i, (tr, te) in enumerate(folds):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        X_tr_n, X_te_n = normalize(X_tr, X_te)

        model = model_class(**model_kwargs)

        t0 = time.perf_counter()
        model.fit(X_tr_n, y_tr)
        train_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        y_pred = model.predict(X_te_n)
        test_times.append(time.perf_counter() - t0)

        m = compute_metrics(y_te, y_pred)
        all_metrics.append(m)

    # Agrega resultados
    result = {}
    for key in ["Acurácia", "Precisão", "Recall", "Especificidade", "F1-Score"]:
        vals = [m[key] for m in all_metrics]
        result[key + "_mean"] = np.mean(vals)
        result[key + "_std"]  = np.std(vals)

    result["T_Treino_mean"] = np.mean(train_times)
    result["T_Teste_mean"]  = np.mean(test_times)
    result["last_metrics"]  = all_metrics[-1]   # Fold final para matriz de confusão
    return result


# =============================================================================
# 4. PERCEPTRON SIMPLES (implementação manual)
# =============================================================================

class Perceptron:
    """
    Perceptron Simples com regra de aprendizado original de Rosenblatt.
    Função de ativação: degrau (threshold em 0).
    Atualização: w ← w + lr * (y - ŷ) * x
    """

    def __init__(self, learning_rate=0.01, n_epochs=100, seed=SEED):
        self.lr       = learning_rate
        self.n_epochs = n_epochs
        self.seed     = seed
        self.weights  = None
        self.bias     = None

    def _activation(self, z):
        return np.where(z >= 0.0, 1, 0)

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        n_samples, n_features = X.shape
        self.weights = rng.normal(0, 0.01, n_features)
        self.bias    = 0.0
        self.errors_ = []

        for _ in range(self.n_epochs):
            errors = 0
            # Embaralha amostras a cada época
            idx = rng.permutation(n_samples)
            for i in idx:
                xi, yi = X[i], y[i]
                z      = np.dot(xi, self.weights) + self.bias
                y_hat  = self._activation(z)
                delta  = self.lr * (yi - y_hat)
                self.weights += delta * xi
                self.bias    += delta
                errors       += int(delta != 0)
            self.errors_.append(errors)
        return self

    def predict(self, X):
        z = X @ self.weights + self.bias
        return self._activation(z)


# =============================================================================
# 5. MLP PARA CLASSIFICAÇÃO (implementação manual)
# =============================================================================

class MLPClassifier:
    """
    Multi-Layer Perceptron para classificação binária.
    - Camadas ocultas com ativação configurável (relu, sigmoid, tanh)
    - Camada de saída com Sigmoid → threshold 0.5
    - Treinamento por backpropagation com SGD mini-batch
    - Inicialização He (ReLU) ou Xavier (sigmoid/tanh)
    """

    def __init__(self, hidden_layers=(64,), activation="relu",
                 learning_rate=0.01, n_epochs=50,
                 batch_size=64, seed=SEED):
        self.hidden_layers = hidden_layers
        self.activation    = activation.lower()
        self.lr            = learning_rate
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size
        self.seed          = seed
        self.weights_      = []
        self.biases_       = []

    # --- Funções de ativação e derivadas ---
    def _act(self, z):
        if self.activation == "relu":
            return np.maximum(0, z)
        if self.activation == "tanh":
            return np.tanh(z)
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))   # sigmoid

    def _act_deriv(self, a):
        if self.activation == "relu":
            return (a > 0).astype(float)
        if self.activation == "tanh":
            return 1 - a ** 2
        return a * (1 - a)   # sigmoid: a já é sigmoid(z)

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    # --- Inicialização ---
    def _init_weights(self, layer_sizes):
        rng = np.random.RandomState(self.seed)
        self.weights_ = []
        self.biases_  = []
        for i in range(len(layer_sizes) - 1):
            fan_in  = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # He para ReLU, Xavier para as demais
            if self.activation == "relu" and i < len(layer_sizes) - 2:
                std = np.sqrt(2.0 / fan_in)
            else:
                std = np.sqrt(2.0 / (fan_in + fan_out))
            W = rng.normal(0, std, (fan_in, fan_out))
            b = np.zeros(fan_out)
            self.weights_.append(W)
            self.biases_.append(b)

    # --- Forward pass ---
    def _forward(self, X):
        activations = [X]
        a = X
        # Camadas ocultas
        for i in range(len(self.weights_) - 1):
            z = a @ self.weights_[i] + self.biases_[i]
            a = self._act(z)
            activations.append(a)
        # Camada de saída (sigmoid binário)
        z_out = a @ self.weights_[-1] + self.biases_[-1]
        a_out = self._sigmoid(z_out)
        activations.append(a_out)
        return activations

    # --- Backward pass (BCE loss) ---
    def _backward(self, activations, y_batch):
        m = y_batch.shape[0]
        grad_w = [None] * len(self.weights_)
        grad_b = [None] * len(self.biases_)

        # Gradiente na saída: d(BCE)/d(a_out) * d(sigmoid)/d(z) = a_out - y
        delta = activations[-1] - y_batch.reshape(-1, 1)   # shape (m,1)

        for i in reversed(range(len(self.weights_))):
            a_prev = activations[i]
            grad_w[i] = (a_prev.T @ delta) / m
            grad_b[i] = delta.mean(axis=0)
            if i > 0:
                delta = (delta @ self.weights_[i].T) * self._act_deriv(activations[i])

        return grad_w, grad_b

    # --- Treino ---
    def fit(self, X, y):
        n_features = X.shape[1]
        layer_sizes = [n_features] + list(self.hidden_layers) + [1]
        self._init_weights(layer_sizes)

        rng = np.random.RandomState(self.seed)
        n   = X.shape[0]

        for epoch in range(self.n_epochs):
            idx = rng.permutation(n)
            X_s, y_s = X[idx], y[idx]

            for start in range(0, n, self.batch_size):
                Xb = X_s[start:start + self.batch_size]
                yb = y_s[start:start + self.batch_size]
                acts = self._forward(Xb)
                gw, gb = self._backward(acts, yb)
                for i in range(len(self.weights_)):
                    self.weights_[i] -= self.lr * gw[i]
                    self.biases_[i]  -= self.lr * gb[i]
        return self

    def predict_proba(self, X):
        acts = self._forward(X)
        return acts[-1].ravel()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# =============================================================================
# 6. EXPERIMENTOS
# =============================================================================

def print_result(name, result):
    m = result
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    print(f"  Acurácia     : {m['Acurácia_mean']:.4f} ± {m['Acurácia_std']:.4f}")
    print(f"  Precisão     : {m['Precisão_mean']:.4f} ± {m['Precisão_std']:.4f}")
    print(f"  Recall       : {m['Recall_mean']:.4f} ± {m['Recall_std']:.4f}")
    print(f"  Especificidade: {m['Especificidade_mean']:.4f} ± {m['Especificidade_std']:.4f}")
    print(f"  F1-Score     : {m['F1-Score_mean']:.4f} ± {m['F1-Score_std']:.4f}")
    print(f"  T. Treino    : {m['T_Treino_mean']:.4f}s")
    print(f"  T. Teste     : {m['T_Teste_mean']:.4f}s")
    lm = m["last_metrics"]
    print_confusion_matrix(lm["TP"], lm["FP"], lm["TN"], lm["FN"])


def run_perceptron_experiments(X, y):
    print("\n" + "="*60)
    print("  PERCEPTRON SIMPLES — Experimentos")
    print("="*60)

    configs = [
        {"learning_rate": 0.1,  "n_epochs": 50},
        {"learning_rate": 0.01, "n_epochs": 50},
        {"learning_rate": 0.01, "n_epochs": 100},
        {"learning_rate": 0.001,"n_epochs": 100},
    ]

    all_results = []
    for cfg in configs:
        name = f"Perceptron  lr={cfg['learning_rate']}  épocas={cfg['n_epochs']}"
        result = run_cv(Perceptron, X, y, cfg)
        print_result(name, result)
        all_results.append((name, result))

    # Melhor configuração
    best = max(all_results, key=lambda x: x[1]["F1-Score_mean"])
    print(f"\n  ★ Melhor Perceptron: {best[0]}")
    print(f"    F1-Score médio: {best[1]['F1-Score_mean']:.4f}")
    return all_results


def run_mlp_experiments(X, y):
    print("\n" + "="*60)
    print("  MLP CLASSIFICAÇÃO — Experimentos")
    print("="*60)

    configs = [
        # (topologia,          ativação,   lr,    épocas)
        {"hidden_layers": (32,),       "activation": "relu",    "learning_rate": 0.01, "n_epochs": 30},
        {"hidden_layers": (64,),       "activation": "relu",    "learning_rate": 0.01, "n_epochs": 30},
        {"hidden_layers": (64,),       "activation": "tanh",    "learning_rate": 0.01, "n_epochs": 30},
        {"hidden_layers": (64,),       "activation": "sigmoid", "learning_rate": 0.01, "n_epochs": 30},
        {"hidden_layers": (64, 32),    "activation": "relu",    "learning_rate": 0.01, "n_epochs": 30},
        {"hidden_layers": (64, 32),    "activation": "tanh",    "learning_rate": 0.01, "n_epochs": 30},
        {"hidden_layers": (128, 64),   "activation": "relu",    "learning_rate": 0.005,"n_epochs": 50},
        {"hidden_layers": (64,),       "activation": "relu",    "learning_rate": 0.001,"n_epochs": 50},
    ]

    all_results = []
    for cfg in configs:
        topo = cfg["hidden_layers"]
        act  = cfg["activation"]
        lr   = cfg["learning_rate"]
        ep   = cfg["n_epochs"]
        name = f"MLP {topo}  act={act}  lr={lr}  épocas={ep}"
        result = run_cv(MLPClassifier, X, y, cfg)
        print_result(name, result)
        all_results.append((name, result))

    best = max(all_results, key=lambda x: x[1]["F1-Score_mean"])
    print(f"\n  ★ Melhor MLP: {best[0]}")
    print(f"    F1-Score médio: {best[1]['F1-Score_mean']:.4f}")
    return all_results


def print_comparative_table(perceptron_results, mlp_results):
    print("\n" + "="*100)
    print("  TABELA COMPARATIVA FINAL")
    print("="*100)
    header = f"{'Modelo':<55} {'Acurácia':>10} {'Precisão':>10} {'Recall':>10} {'Espec.':>10} {'F1-Score':>10} {'T.Treino':>9} {'T.Teste':>9}"
    print(header)
    print("─"*100)

    all_results = perceptron_results + mlp_results
    for name, r in all_results:
        short = name[:54]
        print(
            f"{short:<55} "
            f"{r['Acurácia_mean']:>9.4f}  "
            f"{r['Precisão_mean']:>9.4f}  "
            f"{r['Recall_mean']:>9.4f}  "
            f"{r['Especificidade_mean']:>9.4f}  "
            f"{r['F1-Score_mean']:>9.4f}  "
            f"{r['T_Treino_mean']:>8.3f}s "
            f"{r['T_Teste_mean']:>8.4f}s"
        )
    print("─"*100)


# =============================================================================
# 7. MAIN
# =============================================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  AV3 — Classificação: Perceptron Simples e MLP          ║")
    print("║  Dataset: SBA-Loans-Case-Data-Set (UCI ID: 43539)       ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # 1. Carrega e pré-processa
    X, y = load_and_preprocess(SBA_CSV_PATH)

    # 2. Perceptron Simples
    perceptron_results = run_perceptron_experiments(X, y)

    # 3. MLP Classificação
    mlp_results = run_mlp_experiments(X, y)

    # 4. Tabela comparativa
    print_comparative_table(perceptron_results, mlp_results)

    print("\n✓ Experimentos concluídos.")
