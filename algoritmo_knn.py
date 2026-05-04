import numpy as np 
import urllib.request
import math
from knn_algoritmo import KNN
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
data = np.genfromtxt(urllib.request.urlopen(url), delimiter=",")
print(data)
print(data.shape)
X = data[:, 1:] #entradas
print("Dados de entrada", X)
y = data[:, 0] #saida
print("Dados de saida", y)
#Divisao dos dados em treino(X_train, y_train) (aprendizado da maquina)
#e teste (X_test, y_test) (avaliar o desempenho da maquina)
#Hould Out
def train_test_split(X, y, test_size=0.3, random_state=42): #hould out
    if random_state is not None:
        np.random.seed(random_state)
    if len(X) != len(y):
        raise ValueError("X e y devem ter o mesmo tamanho")
    n_samples = len(X)
    print("Amostra Quantidade", n_samples)
    indices = np.random.permutation(n_samples)
    print("Indices embaralhados", indices)
    n_test = math.ceil(n_samples * test_size) #arredonda para cima
    print("Tamanho da amostra de teste", n_test)
    test_indices = indices[:n_test]
    print("Indices de Teste", test_indices)
    train_indices = indices[n_test:]
    print("Indices de Treino", train_indices)
    if X.ndim == 1:#caso o X seja 1D
        X_train, X_test = X[train_indices], X[test_indices]
    else: #mais dimensoes
        X_train, X_test =  X[train_indices, :], X[test_indices,:]

    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Dados de entrada de treino", X_train)
print("Dados de entrada de teste", X_test)
print("Dados de saida de treino", y_train)
print("Dados de saida de teste", y_test)

modelo = KNN(k=7)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
print("Dados previstos", y_pred) #predito
print("Dados Reais", y_test) #real

#Binario
y_true = [1,0,1,1,0]
y_pred = [1,0,0,1,0]
VP = 2
VN = 2
FP = 1
FN = 0

def accuracy(vp, vn, fp, fn):
    return (vp+vn) / (vp+vn+fp+fn)

def precision(vp, vn, fp, fn):
    return vp / (vp+fp)

def recall(vp, vn, fp, fn):
    return vp / (vp+fn)

def f_score(vp, vn, fp, fn):
    p= precision(vp, vn, fp, fn)
    r = recall(vp, vn, fp, fn)
    return 2* p *r /(p+r)
print(accuracy(VP, VN,FP,FN))
print(precision(VP, VN,FP,FN))
print(recall(VP, VN,FP,FN))
print(f_score(VP, VN,FP,FN))
#Multiclass - metrica acuracia
acertos=0
for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        acertos +=1

accuracy = acertos/len(y_test)
