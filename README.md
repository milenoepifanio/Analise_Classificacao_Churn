# Previsão de Aprovação de Empréstimo Bancário

Este projeto tem como objetivo prever a aprovação de empréstimos bancários com base em características dos solicitantes, utilizando técnicas de análise exploratória e algoritmos de aprendizado supervisionado.

## 🧠 Problema de Negócio

Melhorar o processo de aprovação de crédito pessoal por meio de modelos preditivos, reduzindo decisões manuais e imprecisas na concessão de empréstimos.

## 📚 Dicionário de Dados

O dataset possui 14 colunas e 5.000 registros, com atributos como idade, renda, escolaridade, gastos com cartão e presença de produtos bancários. A variável-alvo é `Personal Loan`, indicando se o empréstimo foi aprovado.


## 📂 Dataset
O dataset é obtido automaticamente via KaggleHub a partir do repositório:

[Bank Loan Approval - Kaggle](https://www.kaggle.com/datasets/vikramamin/bank-loan-approval-lr-dt-rf-and-auc).


# Preparação do projeto

## 📁 Estrutura do Projeto

- `download_data.py`: Script para download e salvamento do dataset diretamente do Kaggle utilizando `kagglehub`.
- `EDA.ipynb`: Notebook de Análise Exploratória de Dados (EDA), com visualizações, limpeza e tratamento da base.
- `Models.ipynb`: Notebook de modelagem, comparação de algoritmos de classificação e avaliação de desempenho.


## 📂 Estrutura de Pastas

```bash
├── data/                      # Contém os dados brutos e processados
│   ├── dados_banco.csv
│   └── dados_banco.parquet
│
├── outputs/                   # Resultados gerados pelo modelo
│   └── resultado_treino_modelos.csv
│
├── scr/                       # Scripts e notebooks do projeto
│   ├── download_data.py       # Script de download dos dados
│   ├── EDA.ipynb              # Análise exploratória de dados
│   └── Models.ipynb           # Modelagem e avaliação
│
├── .gitignore
├── LICENSE.txt
├── README.md
├── pyproject.toml
```

## ⚙️ Tecnologias e Bibliotecas

- Linguagem: Python
- Bibliotecas:
  - pandas: Manipulação de dados em tabelas (DataFrames).
  - numpy: Operações numéricas e vetoriais.
  - seaborn e matplotlib: Visualização de dados (gráficos).
  - scikit-learn: Algoritmos de machine learning e ferramentas de pré-processamento.
  - xgboost, lightgbm: Algoritmos avançados de boosting.
  - joblib: Salvamento e carregamento de modelos.
  - kagglehub: Download de datasets diretamente do Kaggle.


# Exploração e Modelagem

## 📊 Insights da Análise Exploratória (EDA)

- A base é desbalanceada: 90% dos clientes não têm empréstimo aprovado.
- Aprovados têm, em média, **maior renda**, **maior gasto com cartão** e **possuem conta CD**.
- Variáveis-chave: `Income`, `CCAvg`, `CD.Account`, `Education` e `Mortgage`.
- Forte correlação entre `Income`, `CCAvg` e aprovação (`Personal.Loan`).
- Dados com outliers significativos e alguns valores inválidos (ex: experiência negativa).


## 📈 Avaliações Realizadas

- **Acurácia**: Proporção de previsões corretas sobre o total de previsões realizadas.
- **Precisão**: Proporção de positivos previstos que realmente são positivos (foco em evitar falsos positivos).
- **Recall**: Proporção de positivos reais que foram corretamente identificados (foco em evitar falsos negativos).
- **F1-Score**: Média harmônica entre precisão e recall, útil para dados desbalanceados.
- **ROC-AUC**: Mede a capacidade do modelo em distinguir entre classes, variando de 0.5 (aleatório) a 1 (perfeito).


## 🧪 Modelos Utilizados

- **Regressão Logística**.
- **Árvore de Decisão**.
- **Random Forest**.
- **Gradient Boosting**.
- **XGBoost**.
- **LightGBM**.
- **SVM (Support Vector Machine)**.
- **KNN (K-Nearest Neighbors)**.
- **Naive Bayes**.


## Modelagem e Resultados

- **Pré-processamento**:
  - Divisão treino/teste em 80/20.
  - Codificação da variável `Education` via One Hot Encoding.
  - Escalonamento com RobustScaler (mais robusto a outliers).

- **Modelos com melhor desempenho**:
  - **HistGradientBoosting, LightGBM e RandomForest**: Acurácia > 99%, AUC ~0.999.
  - **XGBoost, ExtraTrees e Bagging**: Alta performance e estabilidade.
  - Modelos mais simples (Logistic, NaiveBayes, Ridge) tiveram recall baixo.

- **Variáveis mais importantes**:
  - `Income`: Mais relevante em quase todos os modelos.
  - `CCAvg`: Destaque no LightGBM e RandomForest, menos valorizado no HistGradientBoosting.
  - `Education`: Relevante em alguns modelos, mas impacto variável.

- **Perfil dos algoritmos**:
  - HistGradientBoosting foca em menos variáveis (mais concentrado).
  - LightGBM distribui importância entre mais variáveis (maior sensibilidade).

# Execução e Expectativas

## 📦 Como Executar

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/seu-projeto.git
cd seu-projeto
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute o download do dataset:
```bash
python scr/download_data.py
```

4. Explore os notebooks:
```bash
- notebooks/EDA.ipynb
- notebooks/Models.ipynb
```

# Resultados Esperados

Este projeto busca entender os fatores que influenciam a concessão de crédito e desenvolver modelos preditivos robustos que auxiliem na tomada de decisão automatizada e mais justa.

## Licença

Este projeto está licenciado sob a [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.pt_BR).  
Você pode usar, compartilhar e adaptar, **desde que cite a autoria (Mileno Epifanio)** e **não utilize para fins comerciais**.
