import numpy as np
import csv
import time

# ==========================================
# 1. CLASSE KNN - REGRESSOR (DIST. MANHATTAN)
# ==========================================
class KNN_Regressor_Manhattan:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train #[cite: 2]
        self.y_train = y_train #[cite: 2]

    def predict(self, X_test):
        y_pred = [self._predict(x) for x in X_test] #[cite: 2]
        return np.array(y_pred) #[cite: 2]

    def _predict(self, x):
        # Calcula a distância Manhattan: soma dos valores absolutos das diferenças[cite: 1, 2]
        distancias = [np.sum(np.abs(x - x_treino)) for x_treino in self.X_train]
        
        # Pega os índices dos k vizinhos mais próximos[cite: 2]
        k_indices = np.argsort(distancias)[:self.k]
        
        # Mapeia os valores alvo (y) desses vizinhos[cite: 2]
        k_vizinhos_valores = [self.y_train[i] for i in k_indices]
        
        # PARA REGRESSÃO: O resultado é a MÉDIA dos valores dos vizinhos[cite: 2]
        return np.mean(k_vizinhos_valores)

# ==========================================
# 2. FUNÇÕES PARA CÁLCULO DE MÉTRICAS (REGRESSÃO)
# ==========================================
def calcular_metricas_regressao(y_true, y_pred, num_atributos):
    """
    Calcula o R2-score e o R2-score ajustado manualmente.[cite: 2]
    """
    n = len(y_true) #[cite: 2]
    p = num_atributos #[cite: 2]
    
    # Cálculo do R2-Score[cite: 2]
    media_y_true = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2) #[cite: 2]
    ss_tot = np.sum((y_true - media_y_true) ** 2) #[cite: 2]
    
    if ss_tot == 0:
        r2 = 0.0 #[cite: 2]
    else:
        r2 = 1 - (ss_res / ss_tot) #[cite: 2]
        
    # Cálculo do R2-Score Ajustado[cite: 2]
    if n - p - 1 > 0:
        r2_adj = 1 - ((1 - r2) * (n - 1) / (n - p - 1)) #[cite: 2]
    else:
        r2_adj = 0.0 #[cite: 2]
        
    return r2, r2_adj #[cite: 2]

# ==========================================
# 3. LEITURA DE DADOS E VALIDAÇÃO CRUZADA
# ==========================================
def carregar_dados_sba_regressao(caminho_arquivo, indice_alvo=0):
    linhas_validas = []
    
    with open(caminho_arquivo, 'r', encoding='utf-8') as f: #[cite: 2]
        leitor = csv.reader(f, delimiter=',', quotechar="'", escapechar='\\') #[cite: 2]
        for linha in leitor:
            if not linha or linha[0].startswith('%') or linha[0].startswith('@'): #[cite: 2]
                continue
            linhas_validas.append(linha) #[cite: 2]
            
    if not linhas_validas:
        raise ValueError("Nenhum dado válido encontrado no arquivo.") #[cite: 2]
        
    primeira_linha = linhas_validas[0] #[cite: 2]
    indices_numericos = [] #[cite: 2]
    
    for i in range(len(primeira_linha)): #[cite: 2]
        if i == indice_alvo: 
            continue #[cite: 2]
        try:
            float(primeira_linha[i]) #[cite: 2]
            indices_numericos.append(i) #[cite: 2]
        except ValueError:
            pass 
            
    X = []
    y = []
    
    for linha in linhas_validas: #[cite: 2]
        try:
            alvo = float(linha[indice_alvo].strip()) #[cite: 2]
        except (ValueError, IndexError):
            continue 
            
        atributos = []
        for i in indices_numericos: #[cite: 2]
            if i < len(linha):
                try:
                    valor = linha[i].strip() #[cite: 2]
                    if valor == '?' or valor == '':
                        atributos.append(0.0) #[cite: 2]
                    else:
                        atributos.append(float(valor)) #[cite: 2]
                except ValueError:
                    atributos.append(0.0) #[cite: 2]
            else:
                atributos.append(0.0) #[cite: 2]
        
        X.append(atributos) #[cite: 2]
        y.append(alvo) #[cite: 2]
        
    return np.array(X), np.array(y) #[cite: 2]

def k_fold_cross_validation(X, y, k_folds=5, random_state=42):
    if random_state is not None:
        np.random.seed(random_state) #[cite: 2]
        
    n_samples = len(X) #[cite: 2]
    indices = np.random.permutation(n_samples) #[cite: 2]
    
    tamanhos_folds = np.full(k_folds, n_samples // k_folds, dtype=int) #[cite: 2]
    tamanhos_folds[:n_samples % k_folds] += 1 #[cite: 2]
    
    folds = []
    atual = 0
    for tamanho in tamanhos_folds: #[cite: 2]
        inicio, fim = atual, atual + tamanho #[cite: 2]
        indices_teste = indices[inicio:fim] #[cite: 2]
        indices_treino = np.concatenate((indices[:inicio], indices[fim:])) #[cite: 2]
        folds.append((indices_treino, indices_teste)) #[cite: 2]
        atual = fim #[cite: 2]
        
    return folds #[cite: 2]

# ==========================================
# 4. BLOCO PRINCIPAL DE EXECUÇÃO
# ==========================================
if __name__ == "__main__":
    caminho_dataset_regressao = '../datasets/dataset_interest_rate_46507_regressao' #[cite: 2] 
    
    print("1. Carregando e processando o dataset de REGRESSÃO...")
    X, y = carregar_dados_sba_regressao(caminho_dataset_regressao, indice_alvo=0) #[cite: 2]
    num_atributos = X.shape[1] #[cite: 2]

    print(f"Total de amostras carregadas: {X.shape[0]}") #[cite: 2]
    print(f"Total de atributos (p): {num_atributos}") #[cite: 2]
    
    K_FOLDS = 5 #[cite: 2]
    K_KNN = 5 #[cite: 2]
    
    print(f"\n2. Iniciando Validação Cruzada ({K_FOLDS} Folds) para kNN Regressor (Manhattan, k={K_KNN})...")
    folds = k_fold_cross_validation(X, y, k_folds=K_FOLDS, random_state=42) #[cite: 2]
    
    lista_r2 = [] #[cite: 2]
    lista_r2_adj = [] #[cite: 2]
    lista_tempo_treino = [] #[cite: 2]
    lista_tempo_teste = [] #[cite: 2]
    
    for i, (train_idx, test_idx) in enumerate(folds): #[cite: 2]
        print(f"   -> Executando Fold {i+1}/{K_FOLDS}...") #[cite: 2]
        X_train, y_train = X[train_idx], y[train_idx] #[cite: 2]
        X_test, y_test = X[test_idx], y[test_idx] #[cite: 2]
        
        modelo_knn_reg = KNN_Regressor_Manhattan(k=K_KNN) #[cite: 2]
        
        inicio_treino = time.time() #[cite: 2]
        modelo_knn_reg.fit(X_train, y_train) #[cite: 2]
        fim_treino = time.time() #[cite: 2]
        lista_tempo_treino.append(fim_treino - inicio_treino) #[cite: 2]
        
        inicio_teste = time.time() #[cite: 2]
        predicoes = modelo_knn_reg.predict(X_test) #[cite: 2]
        fim_teste = time.time() #[cite: 2]
        lista_tempo_teste.append(fim_teste - inicio_teste) #[cite: 2]
        
        r2, r2_adj = calcular_metricas_regressao(y_test, predicoes, num_atributos) #[cite: 2]
        lista_r2.append(r2) #[cite: 2]
        lista_r2_adj.append(r2_adj) #[cite: 2]
        
    print("\n" + "="*60)
    print(" RESULTADOS FINAIS - REGRESSÃO MANHATTAN (Média ± Desvio Padrão)")
    print("="*60)
    print(f"R2-Score:           {np.mean(lista_r2):.4f} ± {np.std(lista_r2):.4f}") #[cite: 2]
    print(f"R2-Score Ajustado:  {np.mean(lista_r2_adj):.4f} ± {np.std(lista_r2_adj):.4f}") #[cite: 2]
    print(f"Tempo Treino (s):   {np.mean(lista_tempo_treino):.4f} ± {np.std(lista_tempo_treino):.4f}") #[cite: 2]
    print(f"Tempo Teste (s):    {np.mean(lista_tempo_teste):.4f} ± {np.std(lista_tempo_teste):.4f}") #[cite: 2]
    print("="*60)