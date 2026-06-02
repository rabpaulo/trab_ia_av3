# Defesa Oral - Projeto AV3: Classificação (Perceptron Simples) e Regressão (MLP)

Abaixo estão as perguntas avaliativas elaboradas para arguição da defesa oral, cobrindo tanto a implementação manual do Perceptron Simples (classificação) quanto do MLP (regressão), com foco em métricas, hiperparâmetros, funcionamento dos algoritmos e análise crítica dos resultados.

O objetivo destas questões é avaliar o pensamento crítico do aluno no que diz respeito ao balanço entre desempenho empírico, capacidade de generalização e custos computacionais da solução proposta.

---

## Parte 1 — Classificação: Perceptron Simples

### 1. Escolha dos Hiperparâmetros

* **Pergunta 1.1:** Ao ajustar a taxa de aprendizado e o número de épocas do seu Perceptron Simples no dataset SBA-Loans, como você determinou o ponto em que o modelo parou de melhorar e passou a oscilar sem convergir? Existiu algum critério quantitativo — como monitoramento da taxa de erro por época — para justificar essa parada e evitar tanto o *underfitting* quanto o desperdício computacional?

* **Pergunta 1.2:** O Perceptron Simples só converge garantidamente quando os dados são linearmente separáveis. Dado que o dataset SBA-Loans (previsão de inadimplência) é um problema do mundo real com fronteiras de decisão potencialmente não lineares, como a escolha da taxa de aprendizado impactou a estabilidade do treinamento? Uma taxa muito alta poderia ter impedido a convergência mesmo em regiões aproximadamente separáveis?

### 2. Funcionamento do Algoritmo

* **Pergunta 2.1:** Explique tecnicamente o mecanismo de atualização de pesos do Perceptron Simples. Como a regra de aprendizado do Perceptron difere matematicamente do gradiente descendente estocástico (SGD) utilizado no MLP? Em que situação as duas regras produzem atualizações equivalentes?

* **Pergunta 2.2:** O Perceptron Simples utiliza uma função de ativação degrau (threshold), o que o torna não diferenciável. Como isso impede a aplicação direta do *backpropagation* e limita o modelo a uma única camada? De que forma essa restrição se reflete diretamente nos resultados de classificação observados no seu experimento, especialmente na comparação com o kNN?

### 3. Interpretação dos Resultados

* **Pergunta 3.1:** Nos seus resultados, o kNN (Manhattan) alcançou Acurácia de 88% e F1-Score de 81%, enquanto os modelos Bayesianos atingiram Recall acima de 91% porém com Precisão de apenas 46%. Suponha que o Perceptron Simples apresentasse Recall elevado mas Precisão baixa: como você interpretaria essa divergência no contexto de negócio do problema (aprovação de crédito a pequenas empresas)? Qual métrica seria mais crítica para esse domínio e por quê?

* **Pergunta 3.2:** A matriz de confusão é fundamental para entender onde o modelo erra. No contexto do dataset SBA-Loans, como você diferenciaria o impacto prático de um **Falso Positivo** (prever inadimplência quando o cliente pagaria) de um **Falso Negativo** (prever pagamento quando o cliente daria calote)? Como essa análise justificaria a escolha de um threshold de decisão diferente de 0,5 no Perceptron?

### 4. Diferenças entre os Modelos de Classificação Avaliados

* **Pergunta 4.1:** Compare o Perceptron Simples com o kNN e o Naive Bayes no que diz respeito à natureza do aprendizado: o Perceptron é um modelo paramétrico treinado iterativamente, o kNN é não paramétrico e o Bayes é probabilístico. Como essa diferença fundamental impacta o comportamento de cada modelo em termos de tempo de treinamento, tempo de inferência e capacidade de generalização para novos dados?

* **Pergunta 4.2:** O Bayes Univariado assumiu independência entre atributos e distribuição normal por feature, enquanto o Bayes Multivariado considerou a matriz de covariância completa. Ambos obtiveram Acurácia de 62% — muito abaixo do kNN. De que forma as suposições probabilísticas do Bayes podem ser violadas pela natureza categórica e correlacionada dos atributos do dataset, e como o Perceptron Simples, por ser discriminativo e não generativo, evita essas suposições?

### 5. Análise de Desempenho

* **Pergunta 5.1:** Os modelos Bayesianos apresentaram tempo de teste muito inferior ao kNN (0.007s–0.011s vs. 0.032s–0.033s), pois dispensam o cálculo de distâncias em toda a base de treino. Em que cenário de produção — alta frequência de requisições em tempo real para análise de crédito — você escolheria o Bayes em detrimento do kNN, mesmo aceitando a perda de Precisão de 83% para 46%? Como você defenderia analiticamente esse *trade-off*?

* **Pergunta 5.2:** O Perceptron Simples, por ser um modelo linear de passagem única por amostra, tende a ter custo de inferência O(d), onde d é o número de atributos. Comparado ao kNN com custo O(n·d) por predição (n = tamanho da base), como essa diferença de complexidade computacional se tornaria decisiva em um cenário de *deploy* com milhões de solicitações de crédito por dia? Quais seriam os critérios objetivos para justificar a escolha do modelo mais eficiente?

---

## Parte 2 — Regressão: Multi Layer Perceptron (MLP)

### 1. Escolha dos Hiperparâmetros

* **Pergunta 1.1:** Ao ajustar a taxa de aprendizado e o número de épocas do seu MLP, como você determinou o ponto ótimo onde o modelo deixa de extrair padrões generalizáveis e passa a memorizar o ruído dos dados de treinamento? Existiu algum critério quantitativo para justificar essa parada e prevenir o *overfitting*?

* **Pergunta 1.2:** Considerando as funções de ativação que você implementou (ReLU, Sigmoid, Tanh), explique como a escolha específica de uma dessas funções afeta a dinâmica matemática dos gradientes durante o *backpropagation*. Como essa dinâmica se reflete diretamente na velocidade de convergência do erro e no custo computacional por época? (Observação: nos seus resultados, a Sigmoid apresentou R² de apenas 0,0431 na topologia (20,10), evidenciando o problema de *vanishing gradient* na prática.)

### 2. Funcionamento dos Algoritmos

* **Pergunta 2.1:** Durante a sua implementação manual do *forward pass* e do *backward pass*, explique tecnicamente como o cálculo dos gradientes para a camada de saída difere do cálculo para as camadas ocultas em um problema focado em regressão contínua. Por que a derivada da função de perda (MSE) dita o comportamento inicial dessa propagação reversa?

* **Pergunta 2.2:** O processo de inicialização de pesos é um passo fundamental no treinamento eficiente de redes neurais. Na sua implementação manual, de que forma as estratégias de inicialização de pesos adotadas ajudam a mitigar problemas como o desaparecimento ou explosão de gradientes (*vanishing/exploding gradients*), especialmente nas arquiteturas mais profundas que você testou?

### 3. Interpretação dos Resultados

* **Pergunta 3.1:** Analisando as métricas do seu experimento, suponha que um dos seus modelos apresentou um R² consideravelmente alto, mas o MAE continuou significativamente elevado em relação à escala do domínio do problema. Como você interpretaria essa divergência aparente? Que característica geométrica ou estatística da distribuição dos seus resíduos explicaria esse cenário? (Contexto: a variável alvo Taxa_Juros é categoricamente distribuída em 3 valores, o que já limita estruturalmente o R² máximo atingível.)

* **Pergunta 3.2:** Sabemos que o valor do R² tradicional tende a sempre aumentar, ou ao menos se manter, à medida que adicionamos mais parâmetros (neurônios/camadas) à rede neural. Como você justifica a importância do R² Ajustado na sua análise para penalizar arquiteturas excessivamente complexas e validar a verdadeira eficácia do modelo?

### 4. Diferenças entre as Arquiteturas Avaliadas

* **Pergunta 4.1:** Ao confrontar a arquitetura rasa e larga (50,) contra a profunda e afunilada (20, 10), os seus resultados mostraram que a rede mais larga obteve R² de 0,2958 contra 0,2568 da mais profunda (ambas com ReLU). Como você explica que, para este dataset específico com variável-alvo categórica, redes mais rasas e largas superaram redes mais profundas? Que característica do problema (linearidade, natureza discreta do target) justifica esse comportamento?

* **Pergunta 4.2:** De que forma o aumento na profundidade topológica do seu MLP (adição de uma segunda camada oculta) impactou a correlação entre a redução do erro de validação e o tempo de treinamento requerido? Considerando que (50,) treinou em ~14s e (20,10) com ReLU em ~11.6s, em que ponto exato o ganho métrico marginal deixou de justificar o acréscimo no custo computacional?

### 5. Análise de Desempenho dos Modelos

* **Pergunta 5.1:** Analisando os tempos computacionais de inferência capturados na sua rotina, como a complexidade computacional inerente das diferentes funções de ativação — aliada à densidade das matrizes de peso da topologia escolhida — impactou o *throughput* do seu modelo? Em particular, como a Tanh (que requer exponencial duplo) se compara à ReLU (max simples) no custo por neurônio?

* **Pergunta 5.2:** Se fôssemos realizar o *deploy* da melhor arquitetura do seu MLP (topologia (20,10) com Tanh, R² = 0,2976) em um ambiente de produção com restrições severas de latência, você consideraria estratégico sacrificar frações do RMSE ou R² em prol de uma rede de menor porte — por exemplo, a topologia (10,) com ReLU e R² = 0,2614? Como você defenderia analiticamente a escolha do 'ponto de corte' ideal entre precisão e eficiência de máquina?
