# Draft Amostragem Estratificada

## 1.Contexto

* Número de amostras: 100.000 chips de 256x256 pixels
* Características dos Chips: Cada chip contém várias classes, cada uma ocupando uma certa proporção de área dentro do chip.
* Objetivo da Amostragem: Selecionar um subconjunto representativo de chips que mantenha a distribuição desejada das classes para treinar eficientemente a rede neural.

## 2.Estratégias de Amostragem Estratificada para Chips Multiclasse

2.1 Definir a Distribuição de Classes Desejada

- **Distribuição Real**: Manter a proporção real das classes presente no conjunto total.
- **Distribuição Balanceada**: Equalizar as classes para evitar viés durante o treinamento.
- **Distribuição Específica**: Ajustar as proporções conforme necessidades específicas do projeto ou importância de certas classes.

2.2. Caracterização dos Chips

Calcular a proporção de cada classe dentro de cada chip. Isso pode ser armazenado em uma tabela ou dataframe com as seguintes colunas:

```
{
    {'chip_id': 1, 'Floresta(%)': 70, 'Pasto(%)': 20, 'Água(%)': 5, 'Campo Natural(%)': 5},
    {'chip_id': 2, 'Floresta(%)': 50, 'Pasto(%)': 30, 'Água(%)': 10, 'Campo Natural(%)': 10},
}
```

2.3. Agrupamento de Chips por Perfis de Classes (Clustering)

1. Normalização dos Dados: Certifique-se de que as proporções somam 100% para cada chip.

2. Escolha do Algoritmo de Clustering:
    - K-Means: Para agrupamentos esféricos e bem separados.
    - Hierárquico: Para identificar subgrupos e relações hierárquicas entre clusters.
    - DBSCAN: Para detectar clusters de forma arbitrária e lidar com outliers.

3. Determinação do Número de Clusters:
Utilize métodos como o Elbow Method ou Silhouette Score para definir o número adequado de clusters.
Atribuição de Chips aos Clusters: Cada cluster representará um "estrato" na sua amostragem.

4. Vantagens desta abordagem:
Permite capturar a variabilidade e combinações de classes presentes nos chips.
Facilita a seleção proporcional de chips de cada cluster para a amostra final.
