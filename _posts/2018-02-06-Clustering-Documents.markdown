---
layout: post
title: Clustering Documents and Finding Top 5 Common Terms From Each Cluster
date: 2018-02-06 13:32:20 +0300
description: How to combine (creating a cluster) similar research articles for a topic and collecting most common terms in those similar articles ?. # Add post description (optional)
img: Document-Cluster.png # Add image post (optional)
tags: [KMeans, Document Clustering, Document Similarity, TFIDF, Sklearn, KMeans-Python]
---
How to combine (creating a cluster) similar research articles for a topic and collecting most common terms in those similar articles ?. in this post we will scrape top 10 research articles from pubMed data base for 'Asthma'. We will use different criterial to compare the similarity level among these Dosuments. We will also determine the best possible number of cluster (K) possible for these 10 documents. Then We will cluster them using (K-Means Algorith) and find most common words from these cluster.  

[Detail Code](https://github.com/akpradhn/IAGems/blob/master/Projects/PubmedClusterAnalysis/DocumentClustering7_02_18.md)

[Python Code](https://github.com/akpradhn/IAGems/tree/master/Projects/PubmedClusterAnalysis)

### Objective : Cluster Documents and find top 5 common term from each cluster.

Topic Covered
* Cluster Algorithms
    - K-Means 
* Different types of similarity measures
    - Euclidean distance:
    - Manhattan distance:
    - Cosine similarity:    



### 1. Collecting Data

For this mini project, we are collecting abstracts from first 10 search result for the query 'Asthma' in PubMed database using the API provided on the website. The complete code is available below..

#### Collecting 10 abstracts for the 'Asthma'

First Documents : 

Asthma is a heterogeneous clinical syndrome characterized by airway inflammation, hyper-responsiveness and remodeling.
Airway remodeling is irreversible by current antiasthmatic drugs, and it is the main cause of severe asthma. Airway smooth muscle cells (ASMCs) act as the main effector cells for airway remodeling; the proliferation and hypertrophy of which are involved in airway remodeling. Caveolin (Cav)-1 is present on the surface of ASMCs, which is involved in cell cycle and signal transduction regulation, allowing ASMCs to change from proliferation to apoptosis. The extracellular signal-regulated kinase (ERK)1/2 signaling pathway is a common pathway regulated by various proliferative factors, which demonstrates a regulatory role in airway remodeling of asthma. There have been many studies on the correlation between vasoactive intestinal peptide (VIP) and airway reactivity and inflammation in asthma, but the functions and related mechanisms of ASMCs remain unclear. In this study, we established an airway remodeling model in asthmatic mice, and concluded that VIP inhibits airway remodeling in vivo. The in vitro effect of VIP on interleukin-13-induced proliferation of ASMCs was studied by examining the effects of VIP on expression of ERK1/2, phospho-ERK1/2 and Cav-1 in ASMCs, as well as changes in cell cycle distribution. VIP inhibited phosphorylation of the ERK1/2 signaling pathway and expression of Cav-1 on ASMCs and decreased the proportion of S phase cells in the cell cycle, thus inhibiting the proliferation of ASMCs. This study provides a novel therapeutic mechanism for the treatment of asthma.

### 2. Cleaning Data

Types of Cleaning
* Removing Numerics Data
* Removing Non-ASCII Characters

#### Before Cleanig :

Asthma is a heterogeneous clinical syndrome characterized by airway inflammation, hyper-responsiveness and remodeling.
Airway remodeling is irreversible by current antiasthmatic drugs, and it is the main cause of severe asthma. Airway smooth muscle cells (ASMCs) act as the main effector cells for airway remodeling; the proliferation and hypertrophy of which are involved in airway remodeling. Caveolin (Cav)-1 is present on the surface of ASMCs, which is involved in cell cycle and signal transduction regulation, allowing ASMCs to change from proliferation to apoptosis. The extracellular signal-regulated kinase (ERK)1/2 signaling pathway is a common pathway regulated by various proliferative factors, which demonstrates a regulatory role in airway remodeling of asthma. There have been many studies on the correlation between vasoactive intestinal peptide (VIP) and airway reactivity and inflammation in asthma, but the functions and related mechanisms of ASMCs remain unclear. In this study, we established an airway remodeling model in asthmatic mice, and concluded that VIP inhibits airway remodeling in vivo. The in vitro effect of VIP on interleukin-13-induced proliferation of ASMCs was studied by examining the effects of VIP on expression of ERK1/2, phospho-ERK1/2 and Cav-1 in ASMCs, as well as changes in cell cycle distribution. VIP inhibited phosphorylation of the ERK1/2 signaling pathway and expression of Cav-1 on ASMCs and decreased the proportion of S phase cells in the cell cycle, thus inhibiting the proliferation of ASMCs. This study provides a novel therapeutic mechanism for the treatment of asthma.
     
#### After Clean Text 

Asthma is a heterogeneous clinical syndrome characterized by airway inflammation hyperresponsiveness and remodeling Airway remodeling is irreversible by current antiasthmatic drugs and it is the main cause of severe asthma Airway smooth muscle cells ASMCs act as the main effector cells for airway remodeling the proliferation and hypertrophy of which are involved in airway remodeling Caveolin Cav is present on the surface of ASMCs which is involved in cell cycle and signal transduction regulation allowing ASMCs to change from proliferation to apoptosis The extracellular signalregulated kinase ERK signaling pathway is a common pathway regulated by various proliferative factors which demonstrates a regulatory role in airway remodeling of asthma There have been many studies on the correlation between vasoactive intestinal peptide VIP and airway reactivity and inflammation in asthma but the functions and related mechanisms of ASMCs remain unclear In this study we established an airway remodeling model in asthmatic mice and concluded that VIP inhibits airway remodeling in vivo The in vitro effect of VIP on interleukininduced proliferation of ASMCs was studied by examining the effects of VIP on expression of ERK phosphoERK and Cav in ASMCs as well as changes in cell cycle distribution VIP inhibited phosphorylation of the ERK signaling pathway and expression of Cav on ASMCs and decreased the proportion of S phase cells in the cell cycle thus inhibiting the proliferation of ASMCs This study provides a novel therapeutic mechanism for the treatment of asthma

### 3. Next Step : Document to Vector 

Vector can be a Countervector or a TFIDF Vector

Vector representation of abstract in terms of Term Frequency
![TF]({{site.baseurl}}/assets/img/Doc2vec.png)

Vector representation of abstract in terms of (Term Frequency * Inverse Document Frequency) i.e. (TF-IDF)
![TFIDF]({{site.baseurl}}/assets/img/Doc2vec1.png)

### 3.1 Lets understand how TFIDF is calculated.

The Clean corpus containing 408 words wherein the word 'adult' appears 3 times.
The term frequency (i.e., tf) for cat is then (3 / 408) = 0.007.

Now, we have 10 documents and the word 'adult' appears only in one of these (i.e. 9th Documnets).
Then, the inverse document frequency (i.e., idf) is calculated as log(10 / 1) = 2.303

Thus, the Tf-idf weight for 'adult' is the product of these quantities: 0.03 * 2.303 = 0.016.

In the Next step these weights are then normalised wrt other words in a documents and the final TFIDF matrix can be calculated.

Its clearly upto the choice of user to determine the criteria to asses the similarity between tow documents.

### 4. K-Mean Clustering

Determining Best number of Clusters.  

![Macbook]({{site.baseurl}}/assets/img/output_24_0.png)

The following clusters are assigned to each documents.


| Cluster   | Absracts            |
|-----------|---------------------|
| Cluster 0 | Abstract 5,9 and 10 |
| Cluster 1 | Abstract 3          |
| Cluster 2 | Abstract 1,2 and 6  |
| Cluster 3 | Abstract 8          |
| Cluster 4 | Abstract 7          |
| Cluster 5 | Abstract 4          |


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

* [Different types of similarity measures](http://dataaspirant.com/2015/04/11/five-most-popular-similarity-measures-implementation-in-python/)

* [Finding Number Of Cluster : Wikipedia](https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set?lipi=urn%3Ali%3Apage%3Ad_flagship3_pulse_read%3BGYqlkK3tQ3%2Bb3VWBSvgP8Q%3D%3D)
* [K-means clustering: how it works - Youtube](https://www.youtube.com/watch?v=_aWzGGNrcic&lipi=urn%3Ali%3Apage%3Ad_flagship3_pulse_read%3BGYqlkK3tQ3%2Bb3VWBSvgP8Q%3D%3D)
* [KMeans++ Explained](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)

