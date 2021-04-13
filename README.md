# Stratifying Samples

This python file uses the *Kmeans algorithm to stratify samples of a database*. Its use is simple, you only need to put the columns where you want realizes the stratification, the total of return samples and the sizes of each sample.

T**he stratification is based on distribution of clusters**. For example, if we have 20% of entries in the cluster A, the stratified samples will have nearly 20% of entries of the cluster A.

For the tuning of cluster' numbers we use the *Elbow Method.* The algorithm also realizes the *automatic tranformation of categorical and datetime data to integer numbers and also normalization of data*.

## Normalization

For the normalization we use the Sklearn function, StandardScaler. The data returned of this function will have mean equals to 0 and standart deviation proximally to 1. 

## Algorithm Parameters

To generate the stratified samples you need to following this steps:

<ol>
<li>Assuming that you install the dependencies and this file, you need to create a new object Estratificacao, with the following parameters:
  <ul>
    <li> qtd_base (int) : Quantity of samples.</li>
    <li> tamanho_segmentacao (List) : Length of each sample.</li> 
    <li> percentage (bool Optional) : True if the legth is in percentage values and False otherwise.</li>
    <li> normalize (bool Optional) : True if the data need to be normalized and False otherwise.</li>
    <li> min_clusters (int Optional) : Minimum number of clusters in the Elbow Method.</li>
    <li> max_clusters (int Optional) : Maximum number of clusters in the Elbow Method.</li> 
  </ul>
</li>
<li>With the object Estratification, you only need to call the function estratificar with the following parameters:
  <ul>
    <li> data (DataFrame) : The database that you want stratify samples.</li>
    <li> colunas_segmentacao (List) : Columns used to stratify. </li>
    <li> atributo_diferenciacao (str) : Columns with a different values for each entry.</li>
    <li> colunas_retornadas (List) : Columns returned in the stratified samples. </li>
  </ul>
</li>
