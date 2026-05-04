import numpy as np
class NaiveBayes:
    def fit(self,X_train, y_train):
       n_samples, n_features = X_train.shape
       self._classes = np.unique(y_train)
       n_classes = len(self._classes)
       #calcular media(mean), 
       # variancia(var) e a prior para cada classe
       self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
       self._var = np.zeros((n_classes, n_features), dtype=np.float64)
       self._priors = np.zeros(n_classes, dtype=np.float64)
       for idx, c in enumerate(self._classes): #itera sobre cada classe, loop
           X_c = X_train[y_train == c] #seleciona linhas onde apenas y==c
           self._mean[idx,:] = X_c.mean(axis=0) #calcula a media para cada coluna
           self._var[idx,:] = X_c.var(axis=0) #calcula a variancia para cada coluna
           self._priors[idx] = X_c.shape[0]/ float(n_samples) 
           #calcula a probabilidade a prior da classe c(proporcao das amostras da classe)
    #Probability density function, calcula a probabilidade de x ocorrer em uma distribuicao normal
           #(gaussiana) com media e variancia 
    def _predict(self, x):
        posteriors = []
        for idx, c in enumerate(self._classes): #itera sobre cada classe
                prior = np.log(self._priors[idx])
                posterior = np.sum(np.log(self._pdf(idx, x)))
                posterior = prior + posterior
                posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]
    def predict(self, X_test):
        y_pred = [self._predict(x) for x in X_test]
        return np.array(y_pred)
    def _pdf(self, class_idx, x): #class_idx(indice da classe( 0 nao e 1 sim))
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x-mean)**2)/(2*var))
        denominator = np.sqrt(2* np.pi *var )
        return numerator/denominator
        

   # def predict(self,X_test):
#Exemplo
"""X_train = np.array(
    [[3,2,1],
     [4,5,6],
     [7,8,6],
     [1,3,2],
     [5,4,3]])
y_train = np.array(["Sim", "Nao", "Nao", "Sim", "Sim"])
print(X_train.shape) #2x3
n_samples, n_features = X_train.shape
print(n_samples)
print(n_features)
_classes = np.unique(y_train)
print("Variavel _classes", _classes)
n_classes = len(_classes)
print(n_classes)
mean = np.zeros((n_classes, n_features), dtype=np.float64)
print(mean)
for idx, c in enumerate(_classes):
    print(f"Indices {idx}, Classe {c}")
    X_c = X_train[y_train == c ]
    print(f"X_c (dados da classe) {c}")
    print(X_c)
    mean[idx, :] = X_c.mean(axis=0)
    print(f"Medias: {mean[idx, :]}")"""

X_train = np.array([[1,2],
                   [2,3],
                   [3,4],
                   [6,7],
                   [7,8]])
y_train = np.array([0,0,0,1,1]) 
nb = NaiveBayes() #instancia do algoritmo
nb.fit(X_train, y_train) #treinamento
#Testar
X_test = np.array([[2.5,2.3],
                  [4.5,6.7]]) 
predictions = nb.predict(X_test) #predicao, dados entradas de teste, prever a saida
print(predictions)
