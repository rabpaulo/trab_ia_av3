import numpy as np
import csv
import time

# ==========================================
# 1. CLASSE REGRESSÃO LINEAR MÚLTIPLA
# ==========================================
class RegressaoLinearMultipla:
    def __init__(self):
        self.beta = None

    def fit(self, X, y):
        # Adiciona uma coluna de 1s para o termo de intersecção (bias)
        X_bias = np.c_[np.ones(X.shape[0]), X]
        
        # Equação Normal: beta = (X^T * X)^-1 * X^T * y
        # Usamos a pseudo-inversa (pinv) para maior estabilidade numérica
        xtx = np.dot(X_bias.T, X_bias)
        xtx_inv = np.linalg.pinv(xtx)
        xty = np.dot(X_bias.T, y)
        
        self.beta = np.dot(xtx_inv, xty)

    def predict(self, X):
        X_bias = np.c_[np.ones(X.shape[0]), X]
        return np.dot(X_bias, self.beta)

# ==========================================
# 2. FUNÇÕES DE MÉTRICAS (REGRESSÃO)
# ==========================================
def calcular_metricas(y_true, y_pred, p):
    """
    Calcula exclusivamente o R2-score e o R2-score ajustado.
    """
    n = len(y_true)
    
    media_y = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - media_y) ** 2)
    
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # R2 Ajustado penaliza se houverem muitos atributos (p)
    r2_adj = 1 - ((1 - r2) * (n - 1) / (n - p - 1)) if (n - p - 1) > 0 else 0

    return r2, r2_adj

# ==========================================
# 3. UTILITÁRIOS (CARREGAMENTO E K-FOLD)
# ==========================================
def carregar_dados_regressao(caminho, indice_alvo=0):
    X, y = [], []
    with open(caminho, 'r') as f:
        leitor = csv.reader(f, delimiter=',', quotechar="'", escapechar='\\')
        dados = [l for l in leitor if l and not l[0].startswith(('%', '@'))]
    
    # Filtra colunas numéricas (exceto o alvo)
    indices_num = [i for i in range(len(dados[0])) if i != indice_alvo]
    
    for linha in dados:
        try:
            X.append([float(linha[i]) if linha[i] not in ('?', '') else 0.0 for i in indices_num])
            y.append(float(linha[indice_alvo]))
        except: continue
    return np.array(X), np.array(y)

def k_fold_split(n_samples, k=5):
    indices = np.random.permutation(n_samples)
    fold_sizes = np.full(k, n_samples // k)
    fold_sizes[:n_samples % k] += 1
    current = 0
    for size in fold_sizes:
        test_idx = indices[current:current+size]
        train_idx = np.concatenate([indices[:current], indices[current+size:]])
        yield train_idx, test_idx
        current += size

# ==========================================
# 4. EXECUÇÃO PRINCIPAL
# ==========================================
if __name__ == "__main__":
    # IMPORTANTE: Coloque o caminho correto para o seu dataset
    caminho = '../datasets/dataset_interest_rate_46507_regressao'
    X, y = carregar_dados_regressao(caminho, indice_alvo=0)
    
    # Limita a 10k amostras (usando os primeiros 10k, como preferiu)
    if len(X) > 10000:
        X, y = X[:10000], y[:10000]

    p = X.shape[1] # Número de atributos
    metricas_acumuladas = []

    print(f"Iniciando Regressão Linear Múltipla ({X.shape[0]} amostras)...")
    
    for train_idx, test_idx in k_fold_split(len(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        modelo = RegressaoLinearMultipla()
        
        start_tr = time.time()
        modelo.fit(X_train, y_train)
        t_treino = time.time() - start_tr
        
        start_ts = time.time()
        preds = modelo.predict(X_test)
        t_teste = time.time() - start_ts
        
        r2, r2_adj = calcular_metricas(y_test, preds, p)
        metricas_acumuladas.append([r2, r2_adj, t_treino, t_teste])

    medias = np.mean(metricas_acumuladas, axis=0)
    desvios = np.std(metricas_acumuladas, axis=0)

    print("\n" + "="*60)
    print(" RESULTADOS - REGRESSÃO LINEAR MÚLTIPLA (Média ± DP)")
    print("="*60)
    print(f"R2-Score:          {medias[0]:.4f} ± {desvios[0]:.4f}")
    print(f"R2 Ajustado:       {medias[1]:.4f} ± {desvios[1]:.4f}")
    print(f"Tempo Treino:      {medias[2]:.4f}s")
    print(f"Tempo Teste:       {medias[3]:.4f}s")
    print("="*60)
