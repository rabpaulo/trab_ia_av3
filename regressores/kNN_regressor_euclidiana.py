import numpy as np
import csv
import time

# ==========================================
# 1. CLASSE KNN - REGRESSOR (DIST. EUCLIDIANA)
# ==========================================
class KNN_Regressor_Euclidiana:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = [self._predict(x) for x in X_test]
        return np.array(y_pred)

    def _predict(self, x):
        # Calcula a distância Euclidiana
        distancias = [np.sqrt(np.sum((x - x_treino)**2)) for x_treino in self.X_train]
        
        # Pega os índices dos k vizinhos mais próximos
        k_indices = np.argsort(distancias)[:self.k]
        
        # Mapeia os valores alvo (y) desses vizinhos
        k_vizinhos_valores = [self.y_train[i] for i in k_indices]
        
        # PARA REGRESSÃO: O resultado é a MÉDIA dos valores dos vizinhos
        return np.mean(k_vizinhos_valores)

# ==========================================
# 2. FUNÇÕES PARA CÁLCULO DE MÉTRICAS (REGRESSÃO)
# ==========================================
def calcular_metricas_regressao(y_true, y_pred, num_atributos):
    """
    Calcula o R2-score e o R2-score ajustado manualmente.
    """
    n = len(y_true)
    p = num_atributos
    
    # Cálculo do R2-Score
    media_y_true = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2) # Soma dos quadrados dos resíduos
    ss_tot = np.sum((y_true - media_y_true) ** 2) # Soma total dos quadrados
    
    if ss_tot == 0:
        r2 = 0.0
    else:
        r2 = 1 - (ss_res / ss_tot)
        
    # Cálculo do R2-Score Ajustado
    # Penaliza o R2 se houverem muitos atributos (p) que não ajudam o modelo
    if n - p - 1 > 0:
        r2_adj = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    else:
        r2_adj = 0.0
        
    return r2, r2_adj

# ==========================================
# 3. LEITURA DE DADOS E VALIDAÇÃO CRUZADA
# ==========================================
def carregar_dados_sba_regressao(caminho_arquivo, indice_alvo=0):
    linhas_validas = []
    
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        leitor = csv.reader(f, delimiter=',', quotechar="'", escapechar='\\')
        for linha in leitor:
            if not linha or linha[0].startswith('%') or linha[0].startswith('@'):
                continue
            linhas_validas.append(linha)
            
    if not linhas_validas:
        raise ValueError("Nenhum dado válido encontrado no arquivo.")
        
    primeira_linha = linhas_validas[0]
    indices_numericos = []
    
    for i in range(len(primeira_linha)):
        if i == indice_alvo: 
            continue
        try:
            float(primeira_linha[i])
            indices_numericos.append(i)
        except ValueError:
            pass 
            
    X = []
    y = []
    
    for linha in linhas_validas:
        try:
            # PARA REGRESSÃO: O alvo é lido diretamente como float (valor contínuo)
            alvo = float(linha[indice_alvo].strip())
        except (ValueError, IndexError):
            continue 
            
        atributos = []
        for i in indices_numericos:
            if i < len(linha):
                try:
                    valor = linha[i].strip()
                    if valor == '?' or valor == '':
                        atributos.append(0.0)
                    else:
                        atributos.append(float(valor))
                except ValueError:
                    atributos.append(0.0)
            else:
                atributos.append(0.0)
        
        X.append(atributos)
        y.append(alvo)
        
    return np.array(X), np.array(y)

def k_fold_cross_validation(X, y, k_folds=5, random_state=42):
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    
    tamanhos_folds = np.full(k_folds, n_samples // k_folds, dtype=int)
    tamanhos_folds[:n_samples % k_folds] += 1 
    
    folds = []
    atual = 0
    for tamanho in tamanhos_folds:
        inicio, fim = atual, atual + tamanho
        indices_teste = indices[inicio:fim]
        indices_treino = np.concatenate((indices[:inicio], indices[fim:]))
        folds.append((indices_treino, indices_teste))
        atual = fim
        
    return folds

# ==========================================
# 4. BLOCO PRINCIPAL DE EXECUÇÃO
# ==========================================
if __name__ == "__main__":
    # IMPORTANTE: Aponte para o seu dataset de REGRESSÃO do OpenML
    caminho_dataset_regressao = '../datasets/dataset_interest_rate_46507_regressao'  
    
    print("1. Carregando e processando o dataset de REGRESSÃO...")
    X, y = carregar_dados_sba_regressao(caminho_dataset_regressao, indice_alvo=0) 
    num_atributos = X.shape[1]
    
    LIMITE_AMOSTRAS = 10000
    
    if len(X) > LIMITE_AMOSTRAS:
        print(f"\nLimitando o dataset para as primeiras {LIMITE_AMOSTRAS} amostras...")
        X = X[:LIMITE_AMOSTRAS]
        y = y[:LIMITE_AMOSTRAS]

    print(f"Total de amostras carregadas: {X.shape[0]}")
    print(f"Total de atributos (p): {num_atributos}")
    
    K_FOLDS = 5
    K_KNN = 5
    
    print(f"\n2. Iniciando Validação Cruzada ({K_FOLDS} Folds) para kNN Regressor (Euclidiana, k={K_KNN})...")
    folds = k_fold_cross_validation(X, y, k_folds=K_FOLDS, random_state=42)
    
    lista_r2 = []
    lista_r2_adj = []
    lista_tempo_treino = []
    lista_tempo_teste = []
    
    for i, (train_idx, test_idx) in enumerate(folds):
        print(f"   -> Executando Fold {i+1}/{K_FOLDS}...")
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        modelo_knn_reg = KNN_Regressor_Euclidiana(k=K_KNN)
        
        inicio_treino = time.time()
        modelo_knn_reg.fit(X_train, y_train)
        fim_treino = time.time()
        lista_tempo_treino.append(fim_treino - inicio_treino)
        
        inicio_teste = time.time()
        predicoes = modelo_knn_reg.predict(X_test)
        fim_teste = time.time()
        lista_tempo_teste.append(fim_teste - inicio_teste)
        
        # Calculando as métricas passando o número de atributos (p)
        r2, r2_adj = calcular_metricas_regressao(y_test, predicoes, num_atributos)
        lista_r2.append(r2)
        lista_r2_adj.append(r2_adj)
        
    print("\n" + "="*60)
    print(" RESULTADOS FINAIS - REGRESSÃO EUCLIDIANA (Média ± Desvio Padrão)")
    print("="*60)
    print(f"R2-Score:           {np.mean(lista_r2):.4f} ± {np.std(lista_r2):.4f}")
    print(f"R2-Score Ajustado:  {np.mean(lista_r2_adj):.4f} ± {np.std(lista_r2_adj):.4f}")
    print(f"Tempo Treino (s):   {np.mean(lista_tempo_treino):.4f} ± {np.std(lista_tempo_treino):.4f}")
    print(f"Tempo Teste (s):    {np.mean(lista_tempo_teste):.4f} ± {np.std(lista_tempo_teste):.4f}")
    print("="*60)
