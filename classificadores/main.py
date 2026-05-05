import numpy as np
import time

import csv

def process_sba_data(file_path):
    processed_data = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f, quotechar="'", escapechar='\\')
        for parts in reader:
            if not parts or parts[0].startswith(('%', '@')): continue
            try:
                # Selecionando 11 atributos numéricos + alvo (Default)
                # Colunas: Term, NoEmp, NewExist, CreateJob, RetainedJob, UrbanRural, 
                # DisbursementGross, GrAppv, SBA_Appv, Portion, daysterm, Default
                row = [float(parts[i]) if parts[i] != '?' else 0.0 for i in [11, 12, 13, 14, 15, 17, 22, 26, 27, 30, 32, 34]]
                processed_data.append(row)
            except: continue
    return np.array(processed_data)

def normalize(X):
    return (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10)

def get_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    return acc, prec, rec, f1

class KNN:
    def __init__(self, k=5, dist='euclidean'):
        self.k, self.dist = k, dist
    def fit(self, X, y): self.X_t, self.y_t = X, y
    def predict(self, X_test):
        preds = []
        for x in X_test:
            d = np.sqrt(np.sum((self.X_t - x)**2, axis=1)) if self.dist=='euclidean' else np.sum(np.abs(self.X_t - x), axis=1)
            idx = np.argsort(d)[:self.k]
            preds.append(np.argmax(np.bincount(self.y_t[idx].astype(int))))
        return np.array(preds)

class Bayesian:
    def __init__(self, mode='multivariate'): self.mode = mode
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.params = []
        for c in self.classes:
            X_c = X[y == c]
            self.params.append({'m': np.mean(X_c, axis=0), 'c': np.cov(X_c, rowvar=False) + np.eye(X.shape[1])*1e-4, 'p': len(X_c)/len(X)})
    def predict(self, X_test):
        y_p = []
        for x in X_test:
            probs = []
            for s in self.params:
                if self.mode == 'multivariate':
                    d = len(s['m'])
                    diff = x - s['m']
                    det = np.linalg.det(s['c'])
                    inv = np.linalg.inv(s['c'])
                    prob = (1/((2*np.pi)**(d/2)*np.sqrt(det))) * np.exp(-0.5*diff@inv@diff.T)
                else: # Univariado
                    v = np.diag(s['c'])
                    prob = np.prod(1/np.sqrt(2*np.pi*v) * np.exp(-(x-s['m'])**2/(2*v)))
                probs.append(prob * s['p'])
            y_p.append(self.classes[np.argmax(probs)])
        return np.array(y_p)

def run_experiment(data):
    X, y = normalize(data[:, :-1]), data[:, -1]
    folds = 5
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    f_size = len(X)//folds
    
    models = [("kNN (Euclidiana)", KNN(3, 'euclidean')), ("kNN (Manhattan)", KNN(3, 'manhattan')),
              ("Bayes Univariado", Bayesian('univariate')), ("Bayes Multivariado", Bayesian('multivariate'))]
    
    print(f"{'Classificador':<20} | {'Acurácia':<12} | {'Precisão':<12} | {'Recall':<12} | {'F1-Score':<12} | {'T_Treino':<8} | {'T_Teste':<8}")
    
    for name, model in models:
        m_acc, m_prec, m_rec, m_f1, t_tr, t_ts = [], [], [], [], [], []
        for i in range(folds):
            ts_i = idx[i*f_size:(i+1)*f_size]
            tr_i = np.delete(idx, np.s_[i*f_size:(i+1)*f_size])
            
            start = time.time()
            model.fit(X[tr_i], y[tr_i])
            t_tr.append(time.time() - start)
            
            start = time.time()
            p = model.predict(X[ts_i])
            t_ts.append(time.time() - start)
            
            acc, prec, rec, f1 = get_metrics(y[ts_i], p)
            m_acc.append(acc); m_prec.append(prec); m_rec.append(rec); m_f1.append(f1)
            
        print(f"{name:<20} | {np.mean(m_acc):.2f}±{np.std(m_acc):.2f} | {np.mean(m_prec):.2f}±{np.std(m_prec):.2f} | {np.mean(m_rec):.2f}±{np.std(m_rec):.2f} | {np.mean(m_f1):.2f}±{np.std(m_f1):.2f} | {np.mean(t_tr):.3f}s | {np.mean(t_ts):.3f}s")

data = process_sba_data("dataset_Loans-Case_43539_classificacao")
run_experiment(data)