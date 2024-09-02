# Draft Amostragem Estratificada

1. Contexto

* Número de amostras: 100.000 chips de 256x256 pixels
* Características dos Chips: Cada chip contém várias classes, cada uma ocupando uma certa proporção de área dentro do chip.
* Objetivo da Amostragem: Selecionar um subconjunto representativo de chips que mantenha a distribuição desejada das classes para treinar eficientemente a rede neural.

2. Estratégias de Amostragem Estratificada para Chips Multiclasse

2.1. Definir a Distribuição de Classes Desejada

Primeiramente, é necessário estabelecer qual é a distribuição de classes que você deseja obter no conjunto de treinamento. Isso pode ser baseado em:

- Distribuição Real: Manter a proporção real das classes presente no conjunto total.
- Distribuição Balanceada: Equalizar as classes para evitar viés durante o treinamento.
- Distribuição Específica: Ajustar as proporções conforme necessidades específicas do projeto ou importância de certas classes.