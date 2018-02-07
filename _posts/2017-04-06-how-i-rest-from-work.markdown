---
layout: post
title: Clustering Documents and Finding Top 5 Common Terms From Each Cluster
date: 2018-02-06 13:32:20 +0300
description: How to combine (creating a cluster) similar research articles for a topic and collecting most common terms in those similar articles ?. # Add post description (optional)
img: Document-Cluster.png # Add image post (optional)
tags: [KMeans, Document Clustering, Document Similarity, TFIDF, Sklearn, KMeans-Python]
---
How to combine (creating a cluster) similar research articles for a topic and collecting most common terms in those similar articles ?. in this post we will scrape top 10 research articles from pubMed data base for 'Asthma'. We will use different criterial to compare the similarity level among these Dosuments. We will also determine the best possible number of cluster (K) possible for these 10 documents. Then We will cluster them using (K-Means Algorith) and find most common words from these cluster.  


### Objective : Cluster Documents and find top 5 common term from each cluster.

Topic Covered
* Cluster Algorithms
    - K-Means 
* Different types of similarity measures
    - Euclidean distance:
    - Manhattan distance:
    - Cosine similarity:    


```python
# Final Libraries
from __future__ import print_function
from bs4 import BeautifulSoup
import requests
import re #Cleaning Documents
import pandas as pd #Creating Dataframes
import numpy as np
import random # Freeding Randonness
from sklearn.feature_extraction.text import TfidfVectorizer # For Vectorization
from sklearn.feature_extraction.text import CountVectorizer # Count Vectors
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score # Finding Optimum number of clusters Number 
import matplotlib.pyplot as plt # Visualization
```

### 1. Collecting Data

For this mini project, we are collecting abstracts from first 10 search result for the query 'Asthma' in PubMed database using the API provided on the website. The complete code is available below..


```python
def fetch_abstract(key_word,abst_n):
    
    '''
    Objective:
    ----------
    Returns a list of abstracts of reqired number for a searct quiry from pubmed data base.
    
    Argumnets.
    ----------
    key_word (string): search query text. For e.g. neurodegenerative diseases 
    abst_n  (numeric): Number of Documents required For e.g. 100 
    
    Return:
    -------
    Return a List of abstracts
    
    '''
    '''
    First API to collect PMIDs for abstracts for a search text query
    '''

    payload1 = {'term':key_word, 'retmax' : '15'}
    response1 = requests.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi',params=payload1)
    soup = BeautifulSoup(response1.content, 'html.parser')
    
    print('PMIDs fetched !')
    
    s=[]
    
    ''' 
    Second API to collect abstracts for a required PMID
    '''

    print('Fetching Abstracts...')
    
    for link in soup.find_all('id'):
        
        payload2 = {'db':'pubmed','id':link.string,'rettype':'abstract'}
        tempabs = requests.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi',params = payload2)
        result = BeautifulSoup(tempabs.content, 'html.parser')
    
        try:
            abs= result.find('abstracttext')
            clean=abs.string
            
        except AttributeError:
            pass

        # Store Abstract into a list
        s.append(clean)

        # keeping tract of number of abstract collectd
        if len(s) % 10 == 0:
            progress = len(s)/abst_n * 100
            print(" {}% complete.".format(round(progress, 1)))
                    
    abs_list=s[0:abst_n]
    print(abst_n, ' Abstracts fetched')
    
    return(abs_list)
```

#### Collecting 10 abstracts for the 'Asthma'


```python
key_word = 'Asthma'
abst_n = 10
corpus = fetch_abstract(key_word,abst_n)
```

    PMIDs fetched !
    Fetching Abstracts...
     100.0% complete.
    10  Abstracts fetched



```python
print('Number of Documents in the raw data corpus: ',format(len(corpus)))
print('First 2 Documents :','\n',corpus[0])
```

    Number of Documents in the raw data corpus:  10
    First 2 Documents : 
     Asthma is a heterogeneous clinical syndrome characterized by airway inflammation, hyper-responsiveness and remodeling. Airway remodeling is irreversible by current antiasthmatic drugs, and it is the main cause of severe asthma. Airway smooth muscle cells (ASMCs) act as the main effector cells for airway remodeling; the proliferation and hypertrophy of which are involved in airway remodeling. Caveolin (Cav)-1 is present on the surface of ASMCs, which is involved in cell cycle and signal transduction regulation, allowing ASMCs to change from proliferation to apoptosis. The extracellular signal-regulated kinase (ERK)1/2 signaling pathway is a common pathway regulated by various proliferative factors, which demonstrates a regulatory role in airway remodeling of asthma. There have been many studies on the correlation between vasoactive intestinal peptide (VIP) and airway reactivity and inflammation in asthma, but the functions and related mechanisms of ASMCs remain unclear. In this study, we established an airway remodeling model in asthmatic mice, and concluded that VIP inhibits airway remodeling in vivo. The in vitro effect of VIP on interleukin-13-induced proliferation of ASMCs was studied by examining the effects of VIP on expression of ERK1/2, phospho-ERK1/2 and Cav-1 in ASMCs, as well as changes in cell cycle distribution. VIP inhibited phosphorylation of the ERK1/2 signaling pathway and expression of Cav-1 on ASMCs and decreased the proportion of S phase cells in the cell cycle, thus inhibiting the proliferation of ASMCs. This study provides a novel therapeutic mechanism for the treatment of asthma.


### 2. Cleaning Data

Types of Cleaning
* Removing Numerics Data
* Removing Non-ASCII Characters


```python
def cleanText(raw_text):
    '''
    Objective:
    ----------
    Remove Numeric and Punctuations from a String.
    
    Argument
    --------
    A string.
    
    Reurn
    -------
    A string without numeric/Special character
    
    '''
    #stopword_set = set(stopwords.words("english"))
    return "".join([i for i in re.sub(r'[^\w\s]|\d|','',raw_text)])

clean_corpus = [cleanText(x) for x in corpus]
```


```python
print("Raw Text","\n",corpus[0])
print("Clean Text","\n",clean_corpus[0])
```

    Raw Text 
     Asthma is a heterogeneous clinical syndrome characterized by airway inflammation, hyper-responsiveness and remodeling. Airway remodeling is irreversible by current antiasthmatic drugs, and it is the main cause of severe asthma. Airway smooth muscle cells (ASMCs) act as the main effector cells for airway remodeling; the proliferation and hypertrophy of which are involved in airway remodeling. Caveolin (Cav)-1 is present on the surface of ASMCs, which is involved in cell cycle and signal transduction regulation, allowing ASMCs to change from proliferation to apoptosis. The extracellular signal-regulated kinase (ERK)1/2 signaling pathway is a common pathway regulated by various proliferative factors, which demonstrates a regulatory role in airway remodeling of asthma. There have been many studies on the correlation between vasoactive intestinal peptide (VIP) and airway reactivity and inflammation in asthma, but the functions and related mechanisms of ASMCs remain unclear. In this study, we established an airway remodeling model in asthmatic mice, and concluded that VIP inhibits airway remodeling in vivo. The in vitro effect of VIP on interleukin-13-induced proliferation of ASMCs was studied by examining the effects of VIP on expression of ERK1/2, phospho-ERK1/2 and Cav-1 in ASMCs, as well as changes in cell cycle distribution. VIP inhibited phosphorylation of the ERK1/2 signaling pathway and expression of Cav-1 on ASMCs and decreased the proportion of S phase cells in the cell cycle, thus inhibiting the proliferation of ASMCs. This study provides a novel therapeutic mechanism for the treatment of asthma.
    Clean Text 
     Asthma is a heterogeneous clinical syndrome characterized by airway inflammation hyperresponsiveness and remodeling Airway remodeling is irreversible by current antiasthmatic drugs and it is the main cause of severe asthma Airway smooth muscle cells ASMCs act as the main effector cells for airway remodeling the proliferation and hypertrophy of which are involved in airway remodeling Caveolin Cav is present on the surface of ASMCs which is involved in cell cycle and signal transduction regulation allowing ASMCs to change from proliferation to apoptosis The extracellular signalregulated kinase ERK signaling pathway is a common pathway regulated by various proliferative factors which demonstrates a regulatory role in airway remodeling of asthma There have been many studies on the correlation between vasoactive intestinal peptide VIP and airway reactivity and inflammation in asthma but the functions and related mechanisms of ASMCs remain unclear In this study we established an airway remodeling model in asthmatic mice and concluded that VIP inhibits airway remodeling in vivo The in vitro effect of VIP on interleukininduced proliferation of ASMCs was studied by examining the effects of VIP on expression of ERK phosphoERK and Cav in ASMCs as well as changes in cell cycle distribution VIP inhibited phosphorylation of the ERK signaling pathway and expression of Cav on ASMCs and decreased the proportion of S phase cells in the cell cycle thus inhibiting the proliferation of ASMCs This study provides a novel therapeutic mechanism for the treatment of asthma


### 3. Next Step : Document to Vector 
    Vector can be a Countervector or a TFIDF Vector


```python
countVec = CountVectorizer(stop_words='english')

# Vectors or Term Document Matrix 
tdm_Y = countVec.fit_transform(clean_corpus)

# Get Words from Unique 
unique_words2 = countVec.get_feature_names()

print("The Documents has", len(unique_words2)," words.")

#print(unique_words)
df_Y = pd.DataFrame(tdm_Y.toarray(), columns=unique_words2)
#print(tdm_Y.todense())

display(df_Y.iloc[:,[1,5,10,20,40,100,400]])
```

    The Documents has 408  words.



<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>act</th>
      <th>acute</th>
      <th>adverse</th>
      <th>allergens</th>
      <th>axis</th>
      <th>demonstrates</th>
      <th>vip</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


### 3.1 Lets understand how TFIDF is calculated.

The Clean corpus containing 408 words wherein the word 'adult' appears 3 times.
The term frequency (i.e., tf) for cat is then (3 / 408) = 0.007.

Now, we have 10 documents and the word 'adult' appears only in one of these (i.e. 9th Documnets).
Then, the inverse document frequency (i.e., idf) is calculated as log(10 / 1) = 2.303

Thus, the Tf-idf weight for 'adult' is the product of these quantities: 0.03 * 2.303 = 0.016.

In the Next step these weights are then normalised wrt other words in a documents and the final TFIDF matrix can be calculated.



```python
tfidVec = TfidfVectorizer(stop_words='english')

# Vectors or Term Document Matrix 
tdm_X = tfidVec.fit_transform(clean_corpus)

# Get Words from Unique 
unique_words = tfidVec.get_feature_names()

print("The Documents has", len(unique_words)," words.")

#print(unique_words)
#print(tdm_X.todense())

df_X = pd.DataFrame(tdm_X.toarray(), columns=unique_words)
#type(df_X)
display(df_X.iloc[:,[1,5,10,20,40,100,400]])
```

    The Documents has 408  words.



<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>act</th>
      <th>acute</th>
      <th>adverse</th>
      <th>allergens</th>
      <th>axis</th>
      <th>demonstrates</th>
      <th>vip</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.05488</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.05488</td>
      <td>0.2744</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.329644</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.00000</td>
      <td>0.168225</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.314408</td>
      <td>0.000000</td>
      <td>0.104803</td>
      <td>0.00000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>



```python
cos_sim = cosine_similarity(tdm_X[0:1],tdm_X)
euc_sim = euclidean_distances(tdm_X[0:1],tdm_X)
mah_sim = manhattan_distances(tdm_X[0:1],tdm_X)
```

#### Cosine Similarity


```python
cos_sim
```




    array([[ 1.        ,  0.0255212 ,  0.0154491 ,  0.01334342,  0.08357504,
             0.13946873,  0.00575049,  0.04876417,  0.01491586,  0.10231752]])



Document 1 is similar to Document 6 and 10

#### Euclidean Similarity



```python
euc_sim
```




    array([[ 0.        ,  1.39605071,  1.40324688,  1.40474666,  1.35382787,
             1.31189273,  1.41014149,  1.37930115,  1.40362683,  1.3399123 ]])



This array shows how far the documents vectors are from Document 1. The closer the more similarity.

Its clearly upto the coide of user to determine the criteria to asses the similarity between tow documents.

### 4. K-Mean Clustering

Determining Best number of Clusters 



```python
Clusters = []
Silh_Coefficient = []

for n_cluster in range(2, 10):
    kmeans = KMeans(n_clusters=n_cluster, init='k-means++', max_iter=100, n_init=3).fit(tdm_X)
    label = kmeans.labels_
    sil_coeff = silhouette_score(tdm_X, label, metric='euclidean')
    Clusters.append(n_cluster)
    Silh_Coefficient.append(sil_coeff)
    #print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))

plt.plot(Clusters, Silh_Coefficient)
plt.title('Finding Best Number of Cluster')
plt.xlabel('Number of Cluster')
plt.ylabel('Silhouette_Score')

plt.show()
```


![png](output_24_0.png)



```python
true_k = 6
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=3)
model.fit(tdm_X)

clusters = model.labels_.tolist()
print(clusters)
```

    [2, 2, 1, 5, 0, 2, 4, 3, 0, 0]



```python
print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = tfidVec.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    print("----------------")
    for ind in order_centroids[i, :5]:
        print(" ",terms[ind])
    print("\n")
```

    Top terms per cluster:
    Cluster 0:
    ----------------
      asthma
      endotoxin
      phthalates
      stem
      cells
    
    
    Cluster 1:
    ----------------
      allergens
      years
      allergy
      based
      blomia
    
    
    Cluster 2:
    ----------------
      asthma
      airway
      guidelines
      asmcs
      inflammation
    
    
    Cluster 3:
    ----------------
      lps
      equine
      bronchi
      ngml
      receptors
    
    
    Cluster 4:
    ----------------
      mci
      people
      cognitive
      pa
      population
    
    
    Cluster 5:
    ----------------
      cough
      asthmatic
      efficacy
      longacting
      agonists
    
    


## References: Some Cool Examples


* [Document Clustering with Python](http://brandonrose.org/clustering) (Most Suggested)

* [K-Means clustering on the handwritten digits data](http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html)

* [The Data Science Lab:Clustering With K-Means in Python](https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/)

* [kmeans text clustering](https://pythonprogramminglanguage.com/kmeans-text-clustering/)

* http://ethen8181.github.io/machine-learning/clustering_old/tf_idf/tf_idf.html

* [Different types of similarity measures](http://dataaspirant.com/2015/04/11/five-most-popular-similarity-measures-implementation-in-python/)

* [Finding Number Of Cluster : Wikipedia](https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set?lipi=urn%3Ali%3Apage%3Ad_flagship3_pulse_read%3BGYqlkK3tQ3%2Bb3VWBSvgP8Q%3D%3D)
* [K-means clustering: how it works - Youtube](https://www.youtube.com/watch?v=_aWzGGNrcic&lipi=urn%3Ali%3Apage%3Ad_flagship3_pulse_read%3BGYqlkK3tQ3%2Bb3VWBSvgP8Q%3D%3D)
* [KMeans++ Explained](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)



```python

```
