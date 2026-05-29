# Defesa Oral - Projeto de Regressão com Multi Layer Perceptron (MLP)

Abaixo estão as perguntas avaliativas elaboradas para arguição da defesa oral, focadas na implementação manual do modelo MLP e nas métricas de regressão. 

O objetivo destas questões é avaliar o pensamento crítico do aluno no que diz respeito ao balanço entre o desempenho empírico, a capacidade de generalização e os custos computacionais da solução proposta.

### 1. Escolha dos Hiperparâmetros

* **Pergunta 1.1:** Ao ajustar a taxa de aprendizado e o número de épocas do seu MLP, como você determinou o ponto ótimo onde o modelo deixa de extrair padrões generalizáveis e passa a memorizar o ruído dos dados de treinamento? Existiu algum critério quantitativo para justificar essa parada e prevenir o *overfitting*?
* **Pergunta 1.2:** Considerando as funções de ativação que você implementou (ReLU, Sigmoid, Tanh), explique como a escolha específica de uma dessas funções afeta a dinâmica matemática dos gradientes durante o *backpropagation*. Como essa dinâmica se reflete diretamente na velocidade de convergência do erro e no custo computacional por época?

### 2. Funcionamento dos Algoritmos

* **Pergunta 2.1:** Durante a sua implementação manual do *forward pass* e do *backward pass*, explique tecnicamente como o cálculo dos gradientes para a camada de saída difere do cálculo para as camadas ocultas em um problema focado em regressão contínua. Por que a derivada da função de perda (MSE) dita o comportamento inicial dessa propagação reversa?
* **Pergunta 2.2:** O processo de inicialização de pesos é um passo fundamental no treinamento eficiente de redes neurais. Na sua implementação manual, de que forma as estratégias de inicialização de pesos adotadas ajudam a mitigar problemas como o desaparecimento ou explosão de gradientes (*vanishing/exploding gradients*), especialmente nas arquiteturas mais profundas que você testou?

### 3. Interpretação dos Resultados

* **Pergunta 3.1:** Analisando as métricas do seu experimento, suponha que um dos seus modelos apresentou um R² consideravelmente alto, mas o MAE continuou significativamente elevado em relação à escala do domínio do problema. Como você interpretaria essa divergência aparente? Que característica geométrica ou estatística da distribuição dos seus resíduos explicaria esse cenário?
* **Pergunta 3.2:** Sabemos que o valor do R² tradicional tende a sempre aumentar, ou ao menos se manter, à medida que adicionamos mais parâmetros (neurônios/camadas) à rede neural. Como você justifica a importância do R² Ajustado na sua análise para penalizar arquiteturas excessivamente complexas e validar a verdadeira eficácia do modelo?

### 4. Diferenças entre as Arquiteturas Avaliadas

* **Pergunta 4.1:** Ao confrontar uma arquitetura rasa, porém bastante larga (por exemplo, apenas uma camada com 50 neurônios), contra uma arquitetura profunda e afunilada (por exemplo, 20 neurônios seguidos por 10), quais foram as principais distinções no que tange à captura de mapeamentos não lineares complexos contra a suscetibilidade ao *overfitting*?
* **Pergunta 4.2:** De que forma o aumento na profundidade topológica do seu MLP (adição sucessiva de camadas ocultas) impactou a correlação entre a redução do erro de validação e o tempo de treinamento requerido? Na sua avaliação, em que ponto exato o ganho métrico de desempenho marginal deixou de justificar o acréscimo no custo computacional?

### 5. Análise de Desempenho dos Modelos

* **Pergunta 5.1:** Analisando os tempos computacionais de inferência (teste) capturados na sua rotina, como a complexidade computacional inerente das diferentes funções de ativação — aliada à densidade das matrizes de peso da topologia escolhida — impactou o *throughput* do seu modelo? 
* **Pergunta 5.2:** Se fôssemos realizar o *deploy* da melhor arquitetura do seu MLP em um ambiente de produção que possui restrições severas de latência no tempo de resposta, você consideraria estratégico sacrificar frações do seu RMSE ou R² em prol de uma rede de menor porte? Como você defenderia analiticamente a escolha do 'ponto de corte' ideal entre precisão e eficiência de máquina?
