import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, date
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from IPython.display import display
from IPython.display import HTML

from typing import List, Set, Dict, Tuple, Optional
from typeguard import typechecked
from typing import Union
warnings.filterwarnings("ignore")

class DataNormalize():
    
    def __init__(self):
        """ Classe que contém metódos de normalização e feature scaling necessários para 
            as entradas dos algoritmos de aprendizado.
        """
        pass
    
    @typechecked
    def normalize_func(self, X : Union[pd.DataFrame, pd.Series, np.array]) -> Union[pd.DataFrame, pd.Series, np.array]:
        """Função que recebe uma estrutura de dados e retorna ela normalizada, com média igual à 0
        e desvio padrão igual à 1. 
        
        Args: 
            X (DataFrame) : Pandas Dataframe, Pandas Series ou Numpy Array.
        
        Returns:
            array : Pandas Dataframe, Pandas Series ou Numpy Array normalizado.
        """
        scaler = StandardScaler()

        scaler.fit(X)
        return scaler.transform(X)

class DataTransformNormalize():
    
    @typechecked
    def __init__(self, X : pd.DataFrame, normalize : bool):
        """Classe responsável por conter os métodos de transformações dos dados
        e por chamar os métodos necessários para a normalização das colunas.
        
        Args:
            X (DataFrame) : Pandas DataFrame 
            normalize (bool) : True se os dados precisam ser normalizados e False caso contrário.
        """
        self.X = X
        self.columns_types = dict(self.X.dtypes.astype('str'))
        self.normalize=normalize
    
    @typechecked
    def cat_to_int(self, j : str) -> np.array:
        """Função que converte uma coluna do tipo categorico para números inteiros. 
        
        Args: 
            j (Series) : Nome da coluna com dados categóricos.
        
        Returns:
            array : Pandas Series com dados inteiros.
        """
        le = LabelEncoder()
        return le.fit_transform(self.X[j])

    @typechecked
    def date_to_int(self, j : str) -> np.array:
        """Função que transforma uma coluna datetime em inteiros. Neste metódo, horas,
        minutos e segundos não são considerados na conversão.
        
        Args: 
            j (str) : Nome da coluna com dados datetime.
        
        Returns:
            array : Pandas Series com dados inteiros.
        """
        return 10000*self.X[j].dt.year + 100*self.X[j].dt.month + self.X[j].dt.day
    
    @typechecked
    def transform_data(self) -> pd.DataFrame:
        """Função que transforma e normaliza os dados do DataFrame contido na instanciação do 
        objeto. 
        
        Returns:
            array (DataFrame) : Pandas DataFrame normalizado contendo apenas valores inteiros e reais.
        """
        for i, j in self.columns_types.items():
            if j.lower() in ('object', 'string'):
                self.X[i] = self.cat_to_int(i)
                
            elif j.lower() == 'datetime64[ns]':
                self.X[i] = self.date_to_int(i)
                if self.normalize:
                    self.X[i] = DataNormalize().normalize_func(np.array(self.X[i].values).reshape(-1, 1))
                    
            else:
                if self.normalize:
                    self.X[i] = DataNormalize().normalize_func(np.array(self.X[i].values).reshape(-1, 1))
        return self.X

class Estratificacao():
    
    @typechecked
    def __init__(self, qtd_bases : int, tamanho_segmentacao : List, 
                 percentage = False, normalize = True, 
                 min_clusters=2, max_clusters=15):
        """Classe que recebe uma série de atributos e retorna n bases estratificadas. 
        ou seja, as bases contêm a mesma distribuição de valores nas colunas escolhidas.
        
        Args:
            qtd_base (int) : Quantidade de estratificações que serão geradas.
            tamanho_segmentacao (List) : lista com qtd_base entradas, contendo o tamanho para cada entrada.
            percentage (bool) : True se o tamanho da segmentação é com valores entre 0 e 1 e False caso contrário.
            normalize (bool) : True se os dados vão passar pelo processo de normalização antes e False caso contrário.
            min_clusters (int) : Número mínimo de clusters que será executado o Método de Elbow.
            max_clusters (int) : Número máximo de clusters que será executado o Método de Elbow.
        """
        self.qtd_bases = qtd_bases
        self.tamanho_segmentacao = tamanho_segmentacao
        self.percentage = percentage
        self.normalize = True
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        
    def f_kmeans(self, X):
        """Função que executa o método de Elbow no algorithm K-Means, com os parametros especificados 
        na instanciação da classe. 
        
        Args: 
            X (DataFrame) : Pandas dataFrame contendo apenas valores numéricos.
        
        Returns:
            kmeans : Objeto do Sklearn contendo o K-Means com a melhor configuração de clusters.
        """
        display(HTML(str("<font size=6><b> Modelos </b></font>")))
        display(HTML(str("<font size=4><b> K-Means - Método de Elbow </b></font>")))
        model = KMeans()
        visualizer = KElbowVisualizer(model, k=(self.min_clusters,self.max_clusters))

        visualizer.fit(X)
        visualizer.show()

        return KMeans(n_clusters=visualizer.elbow_value_, random_state=1).fit(X)

    def seg_bases(self, X):
        """Função que divide a base X na quantidade de bases passadas na intanciação. A estratificação
        ocorre através da distribuição dos dados por cluster de saída do K-Means.
        
        Args: 
            X (DataFrame) : Pandas dataFrame contendo apenas valores numéricos.
        
        Returns:
            bases (List) : Uma lista onde cada posição é uma base estratificada com o tamanho 
            previamente especificado.
        """
        self.data['classe'] = self.best_model.predict(X)
        df_temp = self.data

        self.bases = []

        for i in range(self.qtd_bases):
            if self.tamanho_segmentacao[i] < df_temp.shape[0]:
                x_semseg, base_seg, y_train, y_test = train_test_split(df_temp, df_temp.classe,
                                                            test_size=self.tamanho_segmentacao[i]/df_temp.shape[0],
                                                            random_state=0,
                                                            stratify=df_temp.classe)

                df_temp = df_temp[~df_temp[self.atributo_diferenciacao].isin(base_seg[self.atributo_diferenciacao])]
            else:
                base_seg = df_temp
            self.bases.append(base_seg[self.colunas_retornadas])

        return self.bases

    @typechecked
    def estratificar(self, data : pd.DataFrame, 
                     colunas_segmentacao : List, 
                     atributo_diferenciacao : str, 
                     colunas_retornadas : List) -> List:
        
        """Função que estratifica, utilizando as colunas da lista colunas_segmentacao, 
        o DataFrame data em n DataFrames, com tamanhos previamente 
        especificados e contendo apenas os atributos da lista colunas_restornadas.
        
        Args: 
            data (DataFrame) : Pandas DataFrame.
            colunas_segmentacao (List) : Colunas usadas no processo de estratificação.
            atributo_diferenciacao (str) : Coluna com um único valor por entrada.
            colunas_retornadas (List) : Colunas retornadas no fim do processo.
        
        Returns:
            bases (List) : Uma lista onde cada posição é uma base estratificada com o tamanho 
            previamente especificado.
        """
        self.data = data.dropna(subset=colunas_segmentacao)
        self.colunas_segmentacao = colunas_segmentacao
        self.atributo_diferenciacao= atributo_diferenciacao
        self.colunas_retornadas = colunas_retornadas
        
        if self.percentage:
            aux_vector = np.array(self.tamanho_segmentacao)*len(self.data)
            self.tamanho_segmentacao = aux_vector.round()
            
        X = self.data[colunas_segmentacao].copy()
        X =  DataTransformNormalize(X, True).transform_data()
            
        self.best_model = self.f_kmeans(X)

        return self.seg_bases(X)