# Projeto de Previsão de Durabilidade de Pneus em Fórmula 1

Este projeto tem como objetivo prever a durabilidade dos pneus em cada pista do calendário da Fórmula 1, utilizando a biblioteca `FastF1`. A previsão leva em consideração fatores como condições climáticas, tempo de volta e stint.

## Descrição do Projeto

A durabilidade dos pneus é um fator crucial no desempenho de um carro de Fórmula 1. Este projeto visa prever quantas voltas um pneu pode durar em diferentes condições de corrida, utilizando dados históricos fornecidos pela API `FastF1`.

### Funcionalidades

- **Coleta de Dados**: Utiliza a biblioteca `FastF1` para coletar dados históricos e em tempo real de corridas de Fórmula 1.
- **Análise de Condições Climáticas**: Considera as condições climáticas (temperatura, umidade, etc.) para prever a durabilidade dos pneus.
- **Análise de Tempo de Volta**: Analisa o tempo de volta para determinar o desgaste dos pneus.
- **Previsão de Durabilidade**: Prever quantas voltas um pneu pode durar com base nas condições atuais e históricas.

## Requisitos

Para executar este projeto, você precisará das seguintes bibliotecas Python:

- `fastf1`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `Tensor Flow`

## Tree

├── raw_data/\
├   └── f1_dataset.csv\
├── modelos/\
├   └── modelo_pneus.pkl\
└── resultados/\
├   └── graficos/\
└── notebooks/\
├   └── notebook.ipynb/\
├── requirements.txt\
├── setup.py\
├── README.md\
├── Makefile\
