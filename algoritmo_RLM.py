import numpy as np

class RegressionM:
    def __init__(self, X, y):
        self.X = X # entradas
        self.y = y # saida
        self.beta = None

    def fit(self): # Treinamento (descoberta dos parâmetros)
        # 1. Adicionamos uma coluna de 1s à matriz X para representar o intercepto (bias)
        # Isso é necessário para que o modelo calcule o termo constante beta_0
        X_b = np.c_[np.ones((self.X.shape[0], 1)), self.X]
        
        # 2. Aplicamos a Equação Normal: beta = (X^T * X)^-1 * X^T * y
        # Esta é a solução matemática exata para minimizar o erro quadrático em RLM
        # Usamos apenas numpy (np.linalg.inv para inversão e .dot para produto escalar)
        self.beta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(self.y)

    def predict(self, X_test): # Predição
        # 1. Adicionamos a mesma coluna de 1s aos dados de teste para manter a compatibilidade
        X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
        
        # 2. O resultado da predição é o produto escalar entre os dados e os coeficientes beta
        return X_test_b.dot(self.beta)
