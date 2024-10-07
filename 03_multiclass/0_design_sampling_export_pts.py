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
    Método 2


1. Compreendendo o Problema
Dados Totais: 100.000 chips de 256x256 pixels.
Características dos Chips: Cada chip contém várias classes, cada uma ocupando uma certa proporção de área dentro do chip.
Objetivo da Amostragem: Selecionar um subconjunto representativo de chips que mantenha a distribuição desejada das classes para treinar eficientemente a rede neural.


2. Estratégias de Amostragem Estratificada para Chips Multiclasse

2.1. Definir a Distribuição de Classes Desejada
Primeiramente, é necessário estabelecer qual é a distribuição de classes que você deseja obter no conjunto de treinamento.
Isso pode ser baseado em:

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
    1. K-Means: Para agrupamentos esféricos e bem separados.
    2. Hierárquico: Para identificar subgrupos e relações hierárquicas entre clusters.
    3. DBSCAN: Para detectar clusters de forma arbitrária e lidar com outliers.

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
    1. Proporcionalmente: Selecionar número de chips de cada cluster proporcional ao tamanho do cluster no conjunto total.
    2. Equalmente: Selecionar o mesmo número de chips de cada cluster para balancear a representação.
    3. Seleção Aleatória Dentro de Cada Cluster:

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


'''
    Import Session
'''

import sys, os

sys.path.append(os.path.abspath('.'))

import numpy as np
import pandas as pd
import datetime, os
import ee, io
import concurrent, gc
import tensorflow as tf

from retry import retry
from numpy.lib.recfunctions import structured_to_unstructured
from utils.helpers import *
from utils.index import *
from pprint import pprint

PROJECT = 'ee-mapbiomas-imazon'

# ee.Authenticate()
ee.Initialize(project=PROJECT)

'''
    Config Session
'''

PATH_DIR = '/home/jailson/Imazon/dl_applications/source/03_multiclass'

ASSET_REFERENCE = 'projects/ee-mapbiomas-imazon/assets/lulc/reference_map/editted_classification_2020_13'

ASSET_SENTINEL = 'COPERNICUS/S2_HARMONIZED'

ASSET_OUTPUT = 'projects/ee-mapbiomas-imazon/assets/lulc/reference_map/samples_seeds_strat'

SENTINEL_NEW_NAMES = [
    'blue',
    'green',
    'red',
    'red_edge_1',
    'nir',
    'swir1',
    'swir2',
    'pixel_qa'
]

ASSET_IMAGES = {
    's2':{
        'idCollection': ASSET_SENTINEL,
        'bandNames': ['B2', 'B3', 'B4', 'B5', 'B8', 'B11', 'B12', 'QA60'],
        'newBandNames': SENTINEL_NEW_NAMES,
    }
}

NUM_CLASSES = 5

FEATURES = [
    'gv', 
    'npv', 
    'soil', 
    'cloud',
    'gvs',
    'ndfi', 
    'csfi'
]


'''

    Proportion

'''

N_SAMPLES = 3000

PROPORTION = [
    [3, 'SA-23-Y-C',0.6],
    [9,'SA-23-Y-C',0.10],
    [12,'SA-23-Y-C',0.05],
    [15,'SA-23-Y-C',0.26],
    [18,'SA-23-Y-C',0.10],
    [30,'SA-23-Y-C',0.05],
    [24,'SA-23-Y-C',0.05],
    [33,'SA-23-Y-C',0.05]
]

'''

    Request Template    

'''

EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=29)

# image resolution in meters
SCALE = 10

# pre-compute a geographic coordinate system.
proj = ee.Projection('EPSG:4326').atScale(SCALE).getInfo()

# get scales in degrees out of the transform.
SCALE_X = proj['transform'][0]
SCALE_Y = -proj['transform'][4]

# patch size in pixels.
PATCH_SIZE = 256

# offset to the upper left corner.
OFFSET_X = -SCALE_X * PATCH_SIZE / 2
OFFSET_Y = -SCALE_Y * PATCH_SIZE / 2


# request template.
REQUEST = {
      'fileFormat': 'NPY',
      'grid': {
          'dimensions': {
              'width': PATCH_SIZE,
              'height': PATCH_SIZE
          },
          'affineTransform': {
              'scaleX': SCALE_X,
              'shearX': 0,
              'shearY': 0,
              'scaleY': SCALE_Y,
          },
          'crsCode': proj['crs']
      }
  }



'''
    Functions
'''



def serialize(inputs: np.ndarray, labels: np.ndarray) -> bytes:
    features = {
        name: tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(data).numpy()])
        )
        for name, data in {"inputs": inputs, "labels": labels}.items()
    }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()


@retry(delay=0.5)
def get_patch(items):

    """Get a patch centered on the coordinates, as a numpy array."""

    response = {'error': '', 'item': items}
    
    coords = items[1][1]

    image = image_sensor

    if image == None:
        return None, None, None

    request = dict(REQUEST)
    request['expression'] = image
    request['grid']['affineTransform']['translateX'] = coords[0] + OFFSET_X
    request['grid']['affineTransform']['translateY'] = coords[1] + OFFSET_Y


    # criação do objeto Affine usando os parâmetros fornecidos
    # transform = Affine(
    #     request['grid']['affineTransform']['scaleX'], 
    #     request['grid']['affineTransform']['shearX'], 
    #     request['grid']['affineTransform']['translateX'],
    #     request['grid']['affineTransform']['shearY'],
    #     request['grid']['affineTransform']['scaleY'], 
    #     request['grid']['affineTransform']['translateY']
    # )

    # for georeference convertion
    # response['affine'] = transform
    
    try:
        data = np.load(io.BytesIO(ee.data.computePixels(request)))
    except ee.ee_exception.EEException as e:
        response['error']= e
        return None, response, items[0]
    return data, response, items[0]


def export(items, filename):

    future_to_point = {EXECUTOR.submit(get_patch, item): item for item in items}

    # writer = tf.io.TFRecordWriter(filename)

    for future in concurrent.futures.as_completed(future_to_point):
        
        data, label = future.result()

        data = structured_to_unstructured(data)
        label = structured_to_unstructured(label)

        print(data.shape, label.shape)
        
        # serialized = serialize(data, label)

        # writer.write(serialized)
        # writer.flush()
    
    # writer.close()



'''
    Input 
'''

df_proportion = pd.DataFrame(PROPORTION, columns=['classe','grid_name', 'percent'])

roi = ee.Geometry.Polygon([
    [
      [
        -48.00607649166679,
        -4.003006946880755
      ],
      [
        -46.48721662838554,
        -4.003006946880755
      ],
      [
        -46.48721662838554,
        -2.994161602823806
      ],
      [
        -48.00607649166679,
        -2.994161602823806
      ],
      [
        -48.00607649166679,
        -4.003006946880755
      ]
    ]
])

reference_data = ee.Image(ASSET_REFERENCE).rename('label')

# collection = ee.ImageCollection(ASSET_SENTINEL)\
#     .filterDate('2020-05-30', '2020-10-31')\
#     .filterBounds(roi)\
#     .filter('CLOUDY_PIXEL_PERCENTAGE < 30')\
#     .select(ASSET_IMAGES['s2']['bandNames'], ASSET_IMAGES['s2']['newBandNames'])
# 
# collection_w_cloud = remove_cloud_s2(collection)
# 
# collection_w_cloud = collection_w_cloud\
#     .map(lambda image: get_fractions(image))\
#     .map(lambda image: get_ndfi(image))\
#     .map(lambda image: get_csfi(image))
# 
# image_sensor = ee.Image(collection_w_cloud.reduce(ee.Reducer.median())).clip(roi)


'''
    
    Sort Random Samples 

'''

samples_strat = ee.Image(reference_data).stratifiedSample(
    numPoints=N_SAMPLES,
    region=roi,
    scale=10,
    classBand='label',
    dropNulls=True,
    geometries=True,
    classValues=[x[0] for x in PROPORTION],
    classPoints=[int(x[2] * N_SAMPLES) for x in PROPORTION]
)


task = ee.batch.Export.table.toAsset(
    collection=samples_strat,
    description='samples_strat',
    assetId=ASSET_OUTPUT,
)

task.start()

