'''
    Quero fazer uma amostragem estratificada de CHIPs 256 x 256 para treinar uma
    rede neural U-net. No meu conjunto amostral total, tenho 100k chips. Cada chip 
    possue 'n' classes e cada classe tem um valor de área dentro do CHIP.

    O problema é que em uma abordagem tradicional usando somente pontos. Eu teria 
    uma lista de proporção de classes de um mapa, por exemplo: 
        floresta -> 80%, pasto -> 30%, água -> 5%, campo natural -> 5%.
    Ai sortearia aleatoriamente 100 pontos baseado no estrato da proporção das classes. Mas no 
    cenário de chips, cada amostras  é um chip que possue mais de uma classe e também tem sua prórpria
    proporção de classes dentro do chip. Como fazer uma amostragem estratificada nesse cenário? E como representar
    a distribuição de todos os chips (mais de 1k) em um gráfico que seja entendível/?

'''

'''
    Método 1

1. Calcular a Proporção de Classes Dentro de Cada Chip
Para cada chip de 256 x 256, calcule a área de cada classe. 
Isso vai gerar uma proporção interna de classes para cada chip. 

Exemplo:
    Chip A: Floresta 60%, Pasto 30%, Água 5%, Campo Natural 5%.
    Chip B: Floresta 20%, Pasto 50%, Água 15%, Campo Natural 15%.

2. Agrupamento de Chips por Similaridade nas Proporções
Você pode usar métodos de clustering (como k-means ou hierárquico) para agrupar os chips que têm proporções de classes semelhantes. Dessa forma, você vai criar grupos de chips com características internas parecidas em termos de composição das classes.

3. Amostragem Estratificada dos Chips
Dentro de cada grupo de chips (obtido no passo anterior), selecione uma quantidade proporcional de amostras com base nas classes dominantes. Se, por exemplo, em um cluster a maioria dos chips tem alta proporção de floresta, então a amostragem deve refletir isso.
Você pode ajustar a quantidade de chips amostrados em cada cluster para garantir que a diversidade de combinações de classes seja representada.

4. Representar a Distribuição de Classes em Todos os Chips
Para visualizar a distribuição de classes entre todos os chips, você pode criar gráficos como:
Histograma ou Barplot de Proporção de Classes: Mostre a proporção média de cada classe em todos os chips. Isso ajuda a ver a distribuição das classes em nível global.
Heatmap de Proporções: Um heatmap pode ser útil para visualizar a composição de classes dentro de cada chip, com os chips representados no eixo X e as classes no eixo Y, com as cores indicando a proporção da classe.
Boxplot: Para cada classe, exiba a distribuição das proporções de área dentro dos chips. Isso mostrará como cada classe se distribui entre os chips.


'''

from sklearn.cluster import KMeans
import numpy as np

# Exemplo de proporções de classes em 3 chips
chips_proportions = np.array([
    [0.6, 0.3, 0.05, 0.05],  # Chip 1: Floresta, Pasto, Água, Campo Natural
    [0.2, 0.5, 0.15, 0.15],  # Chip 2
    [0.4, 0.4, 0.1, 0.1],    # Chip 3
    # Adicione outros chips aqui
])

# Clusterizar os chips com base nas proporções
kmeans = KMeans(n_clusters=3, random_state=42).fit(chips_proportions)

# Visualizar os clusters
clusters = kmeans.labels_

# Agora você pode fazer a amostragem de chips dentro de cada cluster


'''
    Método 2


1. Compreendendo o Problema
Dados Totais: 100.000 chips de 256x256 pixels.
Características dos Chips: Cada chip contém várias classes, cada uma ocupando uma certa proporção de área dentro do chip.
Objetivo da Amostragem: Selecionar um subconjunto representativo de chips que mantenha a distribuição desejada das classes para treinar eficientemente a rede neural.


2. Estratégias de Amostragem Estratificada para Chips Multiclasse

2.1. Definir a Distribuição de Classes Desejada
Primeiramente, é necessário estabelecer qual é a distribuição de classes que você deseja obter no conjunto de treinamento. Isso pode ser baseado em:

Distribuição Real: Manter a proporção real das classes presente no conjunto total.
Distribuição Balanceada: Equalizar as classes para evitar viés durante o treinamento.
Distribuição Específica: Ajustar as proporções conforme necessidades específicas do projeto ou importância de certas classes.

2.2. Caracterização dos Chips
Calcule e catalogue a proporção de cada classe dentro de cada chip. Isso pode ser armazenado em uma tabela ou dataframe com as seguintes colunas:

Chip_ID	Floresta (%)	Pasto (%)	Água (%)	Campo Natural (%)
1	70	20	5	5
2	50	30	10	10
...	...	...	...	...

2.3. Agrupamento de Chips por Perfis de Classes (Clustering)
Utilize técnicas de clustering para agrupar chips com distribuições de classes semelhantes.

Passos:

Normalização dos Dados: Certifique-se de que as proporções somam 100% para cada chip.
Escolha do Algoritmo de Clustering:
K-Means: Para agrupamentos esféricos e bem separados.
Hierárquico: Para identificar subgrupos e relações hierárquicas entre clusters.
DBSCAN: Para detectar clusters de forma arbitrária e lidar com outliers.
Determinação do Número de Clusters:
Utilize métodos como o Elbow Method ou Silhouette Score para definir o número adequado de clusters.
Atribuição de Chips aos Clusters: Cada cluster representará um "estrato" na sua amostragem.
Vantagens desta abordagem:

Permite capturar a variabilidade e combinações de classes presentes nos chips.
Facilita a seleção proporcional de chips de cada cluster para a amostra final.

2.4. Seleção de Amostras de Cada Cluster
Após agrupar os chips:

Determinar o Tamanho da Amostra Total: Defina quantos chips você deseja na amostra final (e.g., 10.000 chips).
Alocar Amostras por Cluster:
Proporcionalmente: Selecionar número de chips de cada cluster proporcional ao tamanho do cluster no conjunto total.
Equalmente: Selecionar o mesmo número de chips de cada cluster para balancear a representação.
Seleção Aleatória Dentro de Cada Cluster:
Utilize seleção aleatória simples para escolher os chips dentro de cada cluster.
Certifique-se de que a seleção mantém a diversidade interna do cluster.

2.5. Verificação e Ajuste da Distribuição de Classes
Após a seleção:

Calcule a Distribuição Agregada: Some as proporções de classes de todos os chips selecionados para obter a distribuição geral.
Comparação com a Distribuição Desejada: Verifique se a distribuição obtida atende aos objetivos estabelecidos inicialmente.
Ajustes Necessários:
Se houver desvios significativos, ajuste o número de chips selecionados de certos clusters.
Considere incluir ou excluir chips específicos para afinar a distribuição.
3. Visualização da Distribuição de Classes nos Chips
Para representar e entender a distribuição das classes nos chips, você pode utilizar diversas técnicas de visualização.

3.1. Histograma Multiclasse
Descrição: Mostra a distribuição de proporções de cada classe através de histogramas sobrepostos ou separados.
Implementação:
Para cada classe, plote um histograma das proporções através dos chips.
Utilize cores diferentes para cada classe e legendas claras.

3.2. Scatter Plot Multidimensional
Descrição: Representa cada chip como um ponto em um espaço multidimensional, onde cada eixo corresponde à proporção de uma classe.
Implementação:
Para visualização em 2D ou 3D, selecione duas ou três classes principais.
Utilize cores ou formas para indicar clusters ou outras características.
Ferramentas como PCA (Análise de Componentes Principais) podem reduzir a dimensionalidade para facilitar a visualização.

3.3. Boxplot para Comparação de Distribuições
Descrição: Mostra a mediana, quartis e outliers das proporções de cada classe.
Implementação:
Plote um boxplot para cada classe, permitindo comparar facilmente a dispersão e centralidade das proporções.
Útil para identificar classes com alta variabilidade entre os chips.

3.4. Heatmap de Correlação entre Classes
Descrição: Mostra a correlação entre as proporções das diferentes classes nos chips.
Implementação:
Calcule a correlação de Pearson entre as proporções de cada par de classes.
Plote uma matriz de calor onde cores representam o grau de correlação.
Identifica relações como coexistência ou exclusividade entre classes.

3.5. Mapas Auto-Organizáveis (SOM)
Descrição: Técnica de visualização que projeta dados multidimensionais em um espaço bidimensional preservando a topologia dos dados.
Implementação:
Treine um SOM com as proporções de classes dos chips.
O resultado será um mapa onde regiões próximas contêm chips com distribuições de classes similares.
Facilita a identificação de padrões complexos e estruturas nos dados.

4. Ferramentas e Bibliotecas Sugeridas
Python:
Bibliotecas de processamento de dados: pandas, numpy.
Bibliotecas de visualização: matplotlib, seaborn, plotly.
Algoritmos de clustering: scikit-learn.
Técnicas de redução de dimensionalidade: scikit-learn (PCA), somoclu (SOM).
R:
Bibliotecas de visualização: ggplot2, plotly.
Clustering e análise multivariada: cluster, factoextra.
5. Fluxo Resumido do Processo
Coleta e Preparação dos Dados:
Obtenha as proporções de classes de cada chip.
Análise Exploratória Inicial:
Visualize a distribuição geral das classes.
Agrupamento por Clustering:
Agrupe chips com distribuições similares.
Definição da Amostra Estratificada:
Decida o tamanho da amostra e distribua entre os clusters.
Seleção Aleatória de Chips:
Selecione chips dentro de cada cluster.
Verificação da Distribuição Final:
Confirme se a amostra atende às necessidades.
Visualização e Documentação:
Crie gráficos e relatórios que documentem o processo e resultados.
Preparação para o Treinamento da Rede Neural:
Formate e organize os dados selecionados conforme os requisitos da U-Net.

6. Considerações Finais
Balanceamento vs. Representatividade: Sempre considere o trade-off entre ter um conjunto de dados balanceado e representar fielmente a distribuição real dos dados.
Overfitting: Evite selecionar amostras muito homogêneas que possam levar ao overfitting durante o treinamento.
Validação Cruzada: Utilize técnicas de validação cruzada para avaliar a performance da rede com diferentes subconjuntos de dados.
Documentação: Mantenha um registro detalhado do processo de amostragem para replicabilidade e análise futura.
'''