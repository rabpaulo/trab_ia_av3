import numpy as np
import csv
import time

# ==========================================
# 1. CLASSE MLP (MULTI LAYER PERCEPTRON)
# ==========================================
class MLPRegressor:
    def __init__(self, hidden_layers=(10,), learning_rate=0.01, epochs=1000, activation='relu'):
        """
        Inicializa o Multi Layer Perceptron focado em regressão.
        
        :param hidden_layers: Tupla com a quantidade de neurônios por camada oculta.
        :param learning_rate: Taxa de aprendizado.
        :param epochs: Número de épocas de treinamento.
        :param activation: Função de ativação das camadas ocultas ('relu', 'sigmoid', 'tanh').
        """
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation_name = activation
        
        self.weights = []
        self.biases = []
        
        if activation == 'relu':
            self.act_fn = self._relu
            self.act_deriv = self._relu_deriv
        elif activation == 'sigmoid':
            self.act_fn = self._sigmoid
            self.act_deriv = self._sigmoid_deriv
        elif activation == 'tanh':
            self.act_fn = self._tanh
            self.act_deriv = self._tanh_deriv
        else:
            raise ValueError(f"Ativação '{activation}' não suportada.")

    def _relu(self, Z):
        return np.maximum(0, Z)

    def _relu_deriv(self, Z, A):
        return (Z > 0).astype(float)

    def _sigmoid(self, Z):
        Z = np.clip(Z, -500, 500)  # Prevenir overflow
        return 1.0 / (1.0 + np.exp(-Z))

    def _sigmoid_deriv(self, Z, A):
        return A * (1 - A)

    def _tanh(self, Z):
        return np.tanh(Z)

    def _tanh_deriv(self, Z, A):
        return 1 - A**2

    def fit(self, X, y):
        # Para facilitar a vetorização, transpomos X para (n_features, n_samples)
        # e remodelamos y para (1, n_samples)
        A0 = X.T
        Y = y.reshape(1, -1)
        
        n_features = X.shape[1]
        n_samples = X.shape[0]
        
        # Estrutura completa das camadas: entrada + ocultas + saída
        layer_sizes = [n_features] + list(self.hidden_layers) + [1]
        self.weights = []
        self.biases = []
        
        # Inicialização dos pesos e bias
        for i in range(1, len(layer_sizes)):
            # He initialization para ReLU, Xavier para outras
            if self.activation_name == 'relu':
                limit = np.sqrt(2.0 / layer_sizes[i-1])
            else:
                limit = np.sqrt(1.0 / layer_sizes[i-1])
            
            W = np.random.randn(layer_sizes[i], layer_sizes[i-1]) * limit
            b = np.zeros((layer_sizes[i], 1))
            self.weights.append(W)
            self.biases.append(b)

        # Treinamento com Gradiente Descendente em lote (Batch Gradient Descent)
        for epoch in range(self.epochs):
            # --- Forward Pass ---
            A = A0
            caches = []
            
            # Camadas ocultas
            for i in range(len(self.weights) - 1):
                W = self.weights[i]
                b = self.biases[i]
                Z = np.dot(W, A) + b
                A_next = self.act_fn(Z)
                caches.append((A, Z, A_next))
                A = A_next
                
            # Camada de saída (Ativação Linear para Regressão)
            W_out = self.weights[-1]
            b_out = self.biases[-1]
            Z_out = np.dot(W_out, A) + b_out
            A_out = Z_out
            caches.append((A, Z_out, A_out))
            
            # --- Backward Pass ---
            # Derivada da perda (MSE) em relação a A_out
            dZ_next = 2 * (A_out - Y) / n_samples
            
            dW_list = []
            db_list = []
            
            # Gradientes da camada de saída
            A_prev, Z_curr, A_curr = caches[-1]
            dW = np.dot(dZ_next, A_prev.T)
            db = np.sum(dZ_next, axis=1, keepdims=True)
            dW_list.insert(0, dW)
            db_list.insert(0, db)
            
            # Gradientes das camadas ocultas (de trás para frente)
            for i in reversed(range(len(self.weights) - 1)):
                W_next = self.weights[i+1]
                A_prev, Z_curr, A_curr = caches[i]
                
                dA = np.dot(W_next.T, dZ_next)
                dZ = dA * self.act_deriv(Z_curr, A_curr)
                dW = np.dot(dZ, A_prev.T)
                db = np.sum(dZ, axis=1, keepdims=True)
                
                dZ_next = dZ
                dW_list.insert(0, dW)
                db_list.insert(0, db)
                
            # Atualização de pesos e bias
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * dW_list[i]
                self.biases[i] -= self.learning_rate * db_list[i]

    def predict(self, X):
        A = X.T
        
        # Passa pelas camadas ocultas
        for i in range(len(self.weights) - 1):
            W = self.weights[i]
            b = self.biases[i]
            Z = np.dot(W, A) + b
            A = self.act_fn(Z)
            
        # Passa pela camada de saída
        W_out = self.weights[-1]
        b_out = self.biases[-1]
        Z_out = np.dot(W_out, A) + b_out
        
        return Z_out.flatten()

# ==========================================
# 2. FUNÇÕES DE MÉTRICAS (REGRESSÃO)
# ==========================================
def calcular_metricas(y_true, y_pred, p):
    """
    Calcula MSE, RMSE, MAE, R2-Score e R2-Score Ajustado.
    """
    n = len(y_true)
    
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    media_y = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - media_y) ** 2)
    
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # R2 Ajustado penaliza se houver muitos atributos (p)
    r2_adj = 1 - ((1 - r2) * (n - 1) / (n - p - 1)) if (n - p - 1) > 0 else 0

    return mse, rmse, mae, r2, r2_adj

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

def normalizar_features(X_train, X_test):
    # O MLP é sensível a diferentes escalas, normalizamos usando Z-score no conjunto de treino
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1.0  # Evita divisão por zero
    
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    return X_train_norm, X_test_norm

# ==========================================
# 4. EXECUÇÃO PRINCIPAL (EXPERIMENTOS)
# ==========================================
if __name__ == "__main__":
    # IMPORTANTE: Coloque o caminho correto para o seu dataset
    caminho = '../datasets/dataset_interest_rate_46507_regressao'
    
    try:
        X, y = carregar_dados_regressao(caminho, indice_alvo=0)
    except FileNotFoundError:
        print(f"Dataset não encontrado no caminho: {caminho}")
        print("Por favor, atualize o caminho antes de executar ou garanta que o arquivo exista.")
        exit(1)
    
    # Limita a 10k amostras para execução não ser demasiadamente longa
    if len(X) > 10000:
        X, y = X[:10000], y[:10000]

    p = X.shape[1] # Número de atributos

    # Definição de experimentos variando arquitetura, épocas, taxa de aprendizado e ativação
    experimentos = [
        {"hidden_layers": (10,), "lr": 0.01, "epochs": 500, "act": "relu"},
        {"hidden_layers": (20, 10), "lr": 0.05, "epochs": 500, "act": "tanh"},
        {"hidden_layers": (50, 20), "lr": 0.01, "epochs": 300, "act": "relu"},
    ]

    print("="*80)
    print(f" INICIANDO EXPERIMENTOS MLP - {X.shape[0]} amostras, {p} atributos")
    print("="*80)

    resultados = []

    for i, exp in enumerate(experimentos):
        print(f"\n[{i+1}/{len(experimentos)}] Avaliando Arquitetura:")
        print(f"   Camadas Ocultas: {exp['hidden_layers']} | LR: {exp['lr']} | Épocas: {exp['epochs']} | Ativação: {exp['act']}")
        
        metricas_acumuladas = []
        
        for fold, (train_idx, test_idx) in enumerate(k_fold_split(len(X), k=5), 1):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            # Normalização essencial para a estabilidade do gradiente descendente
            X_train_norm, X_test_norm = normalizar_features(X_train, X_test)
            
            modelo = MLPRegressor(
                hidden_layers=exp['hidden_layers'],
                learning_rate=exp['lr'],
                epochs=exp['epochs'],
                activation=exp['act']
            )
            
            start_tr = time.time()
            modelo.fit(X_train_norm, y_train)
            t_treino = time.time() - start_tr
            
            start_ts = time.time()
            preds = modelo.predict(X_test_norm)
            t_teste = time.time() - start_ts
            
            mse, rmse, mae, r2, r2_adj = calcular_metricas(y_test, preds, p)
            metricas_acumuladas.append([mse, rmse, mae, r2, r2_adj, t_treino, t_teste])

        # Calculando as médias e desvios das métricas nos 5 folds
        medias = np.mean(metricas_acumuladas, axis=0)
        desvios = np.std(metricas_acumuladas, axis=0)
        
        resultados.append({
            "config": exp,
            "medias": medias,
            "desvios": desvios
        })

    # Imprimir tabela final
    print("\n\n" + "="*105)
    print(f"{'RESULTADOS FINAIS - MLP REGRESSOR (MÉDIAS DOS FOLDS)':^105}")
    print("="*105)
    print(f"{'Topologia':<15} | {'Ativação':<10} | {'MSE':<12} | {'RMSE':<12} | {'MAE':<10} | {'R2 Score':<10} | {'R2 Ajust':<10} | {'T. Treino (s)':<14}")
    print("-" * 105)
    
    for res in resultados:
        conf = res["config"]
        m = res["medias"]
        topologia_str = str(conf["hidden_layers"])
        ativ_str = conf["act"]
        
        # m = [mse, rmse, mae, r2, r2_adj, t_treino, t_teste]
        linha = f"{topologia_str:<15} | {ativ_str:<10} | {m[0]:<12.4f} | {m[1]:<12.4f} | {m[2]:<10.4f} | {m[3]:<10.4f} | {m[4]:<10.4f} | {m[5]:<14.4f}"
        print(linha)
    
    print("="*105)
