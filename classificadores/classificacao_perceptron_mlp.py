import numpy as np
import csv
import time

SBA_FILE_PATH = "../datasets/dataset_Loans-Case_43539_classificacao"

SEED = 42
np.random.seed(SEED)

#    Colunas selecionadas (índices):
#      11=Term, 12=NoEmp, 13=NewExist, 14=CreateJob, 15=RetainedJob,
#      17=UrbanRural, 22=DisbursementGross, 26=GrAppv, 27=SBA_Appv,
#      30=Portion, 32=daysterm, 34=Default (alvo)

def process_sba_data(file_path):
    processed_data = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f, quotechar="'", escapechar='\\')
        for parts in reader:
            if not parts or parts[0].startswith(('%', '@')):
                continue
            try:
                row = [
                    float(parts[i]) if parts[i] != '?' else 0.0
                    for i in [11, 12, 13, 14, 15, 17, 22, 26, 27, 30, 32, 34]
                ]
                processed_data.append(row)
            except:
                continue
    return np.array(processed_data)


def normalize(X_train, X_test=None):
    """
    Normalização Z-score calculada APENAS no treino.
    Se X_test for None, normaliza X_train com seus próprios parâmetros
    (usado somente para inspecionar os dados).
    """
    mean = np.mean(X_train, axis=0)
    std  = np.std(X_train,  axis=0) + 1e-10
    X_train_n = (X_train - mean) / std
    if X_test is None:
        return X_train_n
    X_test_n = (X_test - mean) / std
    return X_train_n, X_test_n


# =============================================================================
# 2. MÉTRICAS
# =============================================================================

def get_metrics(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    acc  = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return acc, prec, rec, spec, f1, tp, tn, fp, fn


def print_confusion_matrix(tp, tn, fp, fn, indent="  "):
    print(f"{indent}Matriz de Confusão:")
    print(f"{indent}               Previsto 0   Previsto 1")
    print(f"{indent}  Real 0 (Pago)   {tn:>8}     {fp:>8}")
    print(f"{indent}  Real 1 (Calote) {fn:>8}     {tp:>8}")


# =============================================================================
# 3. VALIDAÇÃO CRUZADA K-FOLD
# =============================================================================

def stratified_kfold(y, k=5, seed=SEED):
    """Retorna lista de (train_idx, test_idx) mantendo proporção das classes."""
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    fold_buckets = [[] for _ in range(k)]
    for cls in classes:
        idx = np.where(y == cls)[0]
        idx = rng.permutation(idx)
        for i, split in enumerate(np.array_split(idx, k)):
            fold_buckets[i].extend(split.tolist())
    folds = []
    for i in range(k):
        te = np.array(fold_buckets[i])
        tr = np.array([j for b in (fold_buckets[:i] + fold_buckets[i+1:]) for j in b])
        folds.append((tr, te))
    return folds


def run_cv(model_factory, X, y, k=5):
    folds = stratified_kfold(y, k=k)
    accs, precs, recs, specs, f1s = [], [], [], [], []
    t_trains, t_tests = [], []
    last = {}

    for tr_idx, te_idx in folds:
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        X_tr_n, X_te_n = normalize(X_tr, X_te)

        model = model_factory()

        t0 = time.perf_counter()
        model.fit(X_tr_n, y_tr)
        t_trains.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        y_pred = model.predict(X_te_n)
        t_tests.append(time.perf_counter() - t0)

        acc, prec, rec, spec, f1, tp, tn, fp, fn = get_metrics(y_te, y_pred)
        accs.append(acc); precs.append(prec); recs.append(rec)
        specs.append(spec); f1s.append(f1)
        last = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

    return {
        "acc_m":  np.mean(accs),  "acc_s":  np.std(accs),
        "prec_m": np.mean(precs), "prec_s": np.std(precs),
        "rec_m":  np.mean(recs),  "rec_s":  np.std(recs),
        "spec_m": np.mean(specs), "spec_s": np.std(specs),
        "f1_m":   np.mean(f1s),   "f1_s":   np.std(f1s),
        "t_train": np.mean(t_trains),
        "t_test":  np.mean(t_tests),
        "last":    last,
    }

# 4. PERCEPTRON SIMPLES
#    Regra de Rosenblatt: w ← w + lr * (y - ŷ) * x
#    Ativação: degrau

class Perceptron:
    def __init__(self, learning_rate=0.01, n_epochs=100, seed=SEED):
        self.lr       = learning_rate
        self.n_epochs = n_epochs
        self.seed     = seed

    def _step(self, z):
        return np.where(z >= 0.0, 1, 0)

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        n, d = X.shape
        self.w = rng.normal(0, 0.01, d)
        self.b = 0.0
        for _ in range(self.n_epochs):
            order = rng.permutation(n)
            for i in order:
                z     = X[i] @ self.w + self.b
                y_hat = self._step(z)
                delta = self.lr * (y[i] - y_hat)
                self.w += delta * X[i]
                self.b += delta
        return self

    def predict(self, X):
        return self._step(X @ self.w + self.b)


# 5. MLP PARA CLASSIFICAÇÃO
class MLPClassifier:
    def __init__(self, hidden_layers=(64,), activation="relu",
                 learning_rate=0.01, n_epochs=50, batch_size=64, seed=SEED):
        self.hidden_layers = hidden_layers
        self.activation    = activation.lower()
        self.lr            = learning_rate
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size
        self.seed          = seed

    # ── Ativações ────────────────────────────────────────────────────────────
    def _act(self, z):
        if self.activation == "relu":
            return np.maximum(0.0, z)
        if self.activation == "tanh":
            return np.tanh(z)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))  # sigmoid

    def _act_d(self, a):
        """Derivada em função da ativação já calculada."""
        if self.activation == "relu":
            return (a > 0).astype(float)
        if self.activation == "tanh":
            return 1.0 - a ** 2
        return a * (1.0 - a)  # sigmoid: a = sigmoid(z)

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _init(self, n_in):
        rng = np.random.RandomState(self.seed)
        sizes = [n_in] + list(self.hidden_layers) + [1]
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            std = (np.sqrt(2.0 / fan_in)
                   if self.activation == "relu" and i < len(sizes) - 2
                   else np.sqrt(2.0 / (fan_in + fan_out)))
            self.W.append(rng.normal(0, std, (fan_in, fan_out)))
            self.b.append(np.zeros(fan_out))

    def _forward(self, X):
        acts = [X]
        a = X
        for i in range(len(self.W) - 1):
            a = self._act(a @ self.W[i] + self.b[i])
            acts.append(a)
        a_out = self._sigmoid(a @ self.W[-1] + self.b[-1])
        acts.append(a_out)
        return acts

    def _backward(self, acts, y_batch):
        m   = len(y_batch)
        gW  = [None] * len(self.W)
        gb  = [None] * len(self.b)

        delta = acts[-1] - y_batch.reshape(-1, 1)
        for i in reversed(range(len(self.W))):
            gW[i] = (acts[i].T @ delta) / m
            gb[i] = delta.mean(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * self._act_d(acts[i])
        return gW, gb

    def fit(self, X, y):
        self._init(X.shape[1])
        rng = np.random.RandomState(self.seed)
        n   = len(X)
        for _ in range(self.n_epochs):
            idx = rng.permutation(n)
            Xs, ys = X[idx], y[idx]
            for s in range(0, n, self.batch_size):
                Xb, yb = Xs[s:s+self.batch_size], ys[s:s+self.batch_size]
                acts   = self._forward(Xb)
                gW, gb = self._backward(acts, yb)
                for i in range(len(self.W)):
                    self.W[i] -= self.lr * gW[i]
                    self.b[i] -= self.lr * gb[i]
        return self

    def predict(self, X, threshold=0.5):
        return (self._forward(X)[-1].ravel() >= threshold).astype(int)


def print_result_detail(name, r):
    m = r
    print(f"\n{'─'*62}")
    print(f"  {name}")
    print(f"{'─'*62}")
    print(f"  Acurácia      : {m['acc_m']:.4f} ± {m['acc_s']:.4f}")
    print(f"  Precisão      : {m['prec_m']:.4f} ± {m['prec_s']:.4f}")
    print(f"  Recall        : {m['rec_m']:.4f} ± {m['rec_s']:.4f}")
    print(f"  Especificidade: {m['spec_m']:.4f} ± {m['spec_s']:.4f}")
    print(f"  F1-Score      : {m['f1_m']:.4f} ± {m['f1_s']:.4f}")
    print(f"  T. Treino     : {m['t_train']:.4f}s")
    print(f"  T. Teste      : {m['t_test']:.4f}s")
    l = m["last"]
    print_confusion_matrix(l["tp"], l["tn"], l["fp"], l["fn"])


def print_table(results):
    print("\n" + "=" * 108)
    print("  TABELA COMPARATIVA FINAL — 5-Fold Cross Validation")
    print("=" * 108)
    hdr = (f"{'Modelo':<48} {'Acurácia':>11} {'Precisão':>11} "
           f"{'Recall':>11} {'Espec.':>11} {'F1-Score':>11} {'T.Treino':>9} {'T.Teste':>9}")
    print(hdr)
    print("─" * 108)
    for name, r in results:
        print(
            f"{name[:47]:<48} "
            f"{r['acc_m']:.2f}±{r['acc_s']:.2f}  "
            f"{r['prec_m']:.2f}±{r['prec_s']:.2f}  "
            f"{r['rec_m']:.2f}±{r['rec_s']:.2f}  "
            f"{r['spec_m']:.2f}±{r['spec_s']:.2f}  "
            f"{r['f1_m']:.2f}±{r['f1_s']:.2f}  "
            f"{r['t_train']:.3f}s  "
            f"{r['t_test']:.4f}s"
        )
    print("─" * 108)
    best = max(results, key=lambda x: x[1]["f1_m"])
    print(f"\n  ★ Melhor modelo por F1-Score: {best[0]}")
    print(f"    F1 médio = {best[1]['f1_m']:.4f} | Acurácia = {best[1]['acc_m']:.4f}")

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  AV3 — Classificação: Perceptron Simples e MLP              ║")
    print("║  Dataset: SBA-Loans-Case-Data-Set (UCI ID: 43539)           ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    data = process_sba_data(SBA_FILE_PATH)
    X_all = data[:, :-1]   # 11 features numéricas
    y_all = data[:, -1].astype(int)

    print(f"Amostras carregadas : {len(X_all)}")
    print(f"Features            : {X_all.shape[1]}")
    print(f"Pago (0)  : {(y_all==0).sum()} | Calote (1): {(y_all==1).sum()}\n")

    all_results = []

    print("=" * 62)
    print("  PERCEPTRON SIMPLES")
    print("=" * 62)

    perceptron_configs = [
        dict(learning_rate=0.1,   n_epochs=50),
        dict(learning_rate=0.01,  n_epochs=50),
        dict(learning_rate=0.01,  n_epochs=100),
        dict(learning_rate=0.001, n_epochs=100),
    ]

    for cfg in perceptron_configs:
        label = f"Perceptron  lr={cfg['learning_rate']}  épocas={cfg['n_epochs']}"
        result = run_cv(lambda c=cfg: Perceptron(**c), X_all, y_all)
        print_result_detail(label, result)
        all_results.append((label, result))

    # MLP CLASSIFICAÇÃO
    # Varia: topologia, função de ativação, learning_rate e n_epochs
  
    print("\n" + "=" * 62)
    print("  MLP — CLASSIFICAÇÃO")
    print("=" * 62)

    mlp_configs = [
        dict(hidden_layers=(32,),     activation="relu",    learning_rate=0.01,  n_epochs=30),
        dict(hidden_layers=(64,),     activation="relu",    learning_rate=0.01,  n_epochs=30),
        dict(hidden_layers=(64,),     activation="tanh",    learning_rate=0.01,  n_epochs=30),
        dict(hidden_layers=(64,),     activation="sigmoid", learning_rate=0.01,  n_epochs=30),
        dict(hidden_layers=(64, 32),  activation="relu",    learning_rate=0.01,  n_epochs=30),
        dict(hidden_layers=(64, 32),  activation="tanh",    learning_rate=0.01,  n_epochs=30),
        dict(hidden_layers=(128, 64), activation="relu",    learning_rate=0.005, n_epochs=50),
        dict(hidden_layers=(64,),     activation="relu",    learning_rate=0.001, n_epochs=50),
    ]

    for cfg in mlp_configs:
        topo = cfg["hidden_layers"]
        act  = cfg["activation"]
        lr   = cfg["learning_rate"]
        ep   = cfg["n_epochs"]
        label = f"MLP {str(topo):<12} act={act:<8} lr={lr}  épocas={ep}"
        result = run_cv(lambda c=cfg: MLPClassifier(**c), X_all, y_all)
        print_result_detail(label, result)
        all_results.append((label, result))

    # ── Tabela final ──────────────────────────────────────────────────────────
    print_table(all_results)

    print("\n✓ Experimentos concluídos.")
