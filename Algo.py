# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:29:36 2017

@author: lulu
"""

def tuning_arbre(nom_tuning,critère_de_tuning,parameters,type_pretraitement,type_pred):

    classifier = DecisionTreeClassifier()
    gridsearch = GridSearchCV(estimator = classifier, param_grid = parameters,scoring = critère_de_tuning ,cv = 10)
    
    gridsearch = gridsearch.fit(xtrain,ytrain)
    gs_best_parameters_tree_classifier = gridsearch.best_params_
    
    #precision = precision_score()
    criterion = gs_best_parameters_tree_classifier["criterion"]
    max_depth = gs_best_parameters_tree_classifier["max_depth"]
    max_features = gs_best_parameters_tree_classifier["max_features"]
    
    classifier = DecisionTreeClassifier(criterion = criterion,
                                        max_depth = max_depth ,
                                        max_features = max_features)
    classifier.fit(xtrain,ytrain)
    
    feature_importance = classifier.feature_importances_
    feature_importance_data_frame = pd.DataFrame(data = column_stack([columns,feature_importance]))
    feature_importance_data_frame.sort_values
    
    ypred = classifier.predict(xtest)
    accuracy = accuracy_score(ytest,ypred)
    precision = precision_score(ytest,ypred)
    rappel = precision_score(ytest,ypred)
    MCC = matthews_corrcoef(ytest,ypred)
    roc_auc = roc_auc_score(ytest,ypred)

    ##Placement des résultats dans le dictionnaire##
    
    res_pred[type_pred][type_pretraitement][nom_tuning] = {}
    res_pred[type_pred][type_pretraitement][nom_tuning]["arbre"] = {}

    res_pred[type_pred][type_pretraitement][nom_tuning]["arbre"]["accuracy"] = accuracy
    res_pred[type_pred][type_pretraitement][nom_tuning]["arbre"]["precision"] = precision
    res_pred[type_pred][type_pretraitement][nom_tuning]["arbre"]["rappel"] = rappel
    res_pred[type_pred][type_pretraitement][nom_tuning]["arbre"]["MCC"] = MCC
    res_pred[type_pred][type_pretraitement][nom_tuning]["arbre"]["roc_auc"] = roc_auc


    return feature_importance_data_frame

nom_tuning = "tuning_accuracy"
critère_de_tuning = "accuracy"
type_pretraitement = "Prétraitement_basique"
type_pred = "Prediction_label_unique"



def tuning_RDF(nom_tuning,critère_de_tuning,parameters,type_pretraitement,type_pred):

    classifier = RandomForestClassifier()
    gridsearch = GridSearchCV(estimator = classifier, param_grid = parameters,scoring = critère_de_tuning ,cv = 10)
    ytrain_rdf = ytrain.values.ravel()
    
    gridsearch = gridsearch.fit(xtrain,ytrain_rdf)
    gs_best_parameters_rdf_classifier = gridsearch.best_params_
    
    #precision = precision_score()
    n_estimators = gs_best_parameters_rdf_classifier["n_estimators"]
    criterion = gs_best_parameters_rdf_classifier["criterion"]
    max_depth = gs_best_parameters_rdf_classifier["max_depth"]
    max_features = gs_best_parameters_rdf_classifier["max_features"]
    
    classifier = RandomForestClassifier(n_estimators = n_estimators,
                                        criterion = criterion,
                                        max_depth = max_depth ,
                                        max_features = max_features)
    classifier.fit(xtrain,ytrain_rdf)
    
    feature_importance = classifier.feature_importances_
    feature_importance_data_frame = pd.DataFrame(data = column_stack([columns,feature_importance]))
    feature_importance_data_frame.sort_values
    
    ypred = classifier.predict(xtest)
    accuracy = accuracy_score(ytest,ypred)
    precision = precision_score(ytest,ypred)
    rappel = precision_score(ytest,ypred)
    MCC = matthews_corrcoef(ytest,ypred)
    roc_auc = roc_auc_score(ytest,ypred)

    ##Placement des résultats dans le dictionnaire##
    
    res_pred[type_pred][type_pretraitement][nom_tuning] = {}
    res_pred[type_pred][type_pretraitement][nom_tuning]["rdf"] = {}
    res_pred[type_pred][type_pretraitement][nom_tuning]["rdf"]["accuracy"] = accuracy
    res_pred[type_pred][type_pretraitement][nom_tuning]["rdf"]["precision"] = precision
    res_pred[type_pred][type_pretraitement][nom_tuning]["rdf"]["rappel"] = rappel
    res_pred[type_pred][type_pretraitement][nom_tuning]["rdf"]["MCC"] = MCC
    res_pred[type_pred][type_pretraitement][nom_tuning]["rdf"]["roc_auc"] = roc_auc


    return feature_importance_data_frame




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##Suppression des données ##

def contain(list,elem):
    for item in list:
        if(item == elem):
            return True
    return False

df = pd.read_csv("donnees-defi-egc.csv" ,sep=",",decimal=".")

newdf = df.replace('?',np.nan)
res = newdf.apply(lambda x: sum(x.isnull().values), axis = 0)

columnToSuppress = []

for i in range(res.size):
   if (res[i] > df.shape[0]/2):
       columnToSuppress.append(i)



ColumnToModify = []
res = newdf.apply(lambda x: sum(x.isnull().values), axis = 0)


df = newdf.drop(newdf.columns[columnToSuppress],axis = 1)

columnToModify = []

for i in range(res.size):
    if (res[i] > 0):
        if(not (contain(columnToSuppress,i))):
#            print(newdf.columns[i])
#            print(res[i])
#            print()
            columnToModify.append(i)
            

inds = pd.isnull(df["ESPECE"]).nonzero()[0]  
df = df.drop(inds)
df.reset_index(drop = True)

##Etude des attributs##

res = np.unique(df["ESPECE"])
res.shape[0]

#columns = df.columns
#for column in columns:
#    print(column)
#len(columns)

##ANNEEREALISATIONDIAG##
df.hist()

inds = pd.isnull(df["ANNEEREALISATIONDIAGNOSTIC"]).nonzero()[0]  
df = df.drop(inds)
df.reset_index(drop = True)


summary_temp_avg = pd.DataFrame(df.groupby('ANNEEREALISATIONDIAGNOSTIC')['DEFAUT','Collet','Houppier','Racine','Tronc'].mean())
summary_temp_avg.plot(kind='bar')

#df.dropna(subset = ['ANNEEREALISATIONDIAGNOSTIC'])


##ESPECE##



summary_espece_avg = pd.DataFrame(df.groupby('ESPECE')['DEFAUT','Collet','Houppier','Racine','Tronc'].mean())
summary_espece_sum = pd.DataFrame(df.groupby('ESPECE')['DEFAUT','Collet','Houppier','Racine','Tronc'].sum())
summary_espece_count = pd.DataFrame(df.groupby('ESPECE')['DEFAUT','Collet','Houppier','Racine','Tronc'].count())




summary_espece_avg["nombre de défaut"] = summary_espece_sum["DEFAUT"]
summary_espece_avg["nombre de valeur"] = summary_espece_count["DEFAUT"]



columns = ['ADR_SECTEUR', 'ANNEEDEPLANTATION']


df = df.drop( ["ANNEETRAVAUXPRECONISESDIAG","TRAVAUXPRECONISESDIAG" ,"CODE_PARENT_DESC"],axis = 1)
df = df.dropna()

df.reset_index(drop = True)



    
##nombre de défaut##
summary_defaut_count = pd.DataFrame(df.groupby('DEFAUT')['DEFAUT'].count())


    

####MACHINE LEARNING####
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df = df.drop(["NOTEDIAGNOSTIC"],axis = 1)


column = ["CODE","CODE_PARENT","DIAMETREARBREAUNMETRE","ESPECE","FREQUENTATIONCIBLE","GENRE_BOTA","NOTEDIAGNOSTIC","PRIORITEDERENOUVELLEMENT","SOUS_CATEGORIE","SOUS_CATEGORIE_DESC","STADEDEDEVELOPPEMENT","STADEDEVELOPPEMENTDIAG","TROTTOIR","VIGUEUR"]

for column1 in column:
    df[column1] = le.fit_transform(df[column1])

##Dictionnaire regroupant tous les résultats##
res_pred = {}

df.to_csv("donnéesNetoyés",sep = ",")

####PREDICTION LABEL UNIQUE####
res_pred["Prediction_label_unique"] = {}

y = df[["DEFAUT"]]
x = df.drop(["DEFAUT","Collet","Houppier","Racine","Tronc"],axis = 1)
columns = x.columns

##Prediction Arbre##

xtrain, xtest,ytrain, ytest = train_test_split( x, y, test_size=0.33, random_state=42)

tree_classifier = DecisionTreeClassifier()

##Prediction avec prétraitement_basique##
res_pred["Prediction_label_unique"]["Prétraitement_basique"] = {}



from sklearn.model_selection import GridSearchCV

##Tuning##

parameters = {'criterion' : ['gini','entropy'], 'max_depth' : [2,5,7,10,12,15,18],'max_features' : [3,4,5,6,7,'auto'] }



feature_importance_data_frame = tuning_arbre("tuning_accuracy","accuracy",parameters,"Prétraitement_basique","Prediction_label_unique")
feature_importance_data_frame = tuning_arbre("tuning_f1","f1",parameters,"Prétraitement_basique","Prediction_label_unique")
feature_importance_data_frame = tuning_arbre("tuning_log_loss","neg_log_loss",parameters,"Prétraitement_basique","Prediction_label_unique")
feature_importance_data_frame = tuning_arbre("tuning_log_error","neg_mean_squared_log_error",parameters,"Prétraitement_basique","Prediction_label_unique")
feature_importance_data_frame = tuning_arbre("tuning_roc","roc_auc",parameters,"Prétraitement_basique","Prediction_label_unique")

##Affichage Arbre##

#from sklearn.externals.six import StringIO  
#from IPython.display import Image  
#from sklearn.tree import export_graphviz
#import pydotplus
#
#dot_data = StringIO()
#export_graphviz(tree_classifier, out_file=dot_data,  
#                filled=True, rounded=True,
#                special_characters=True)
#
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#Image(graph.create_png())

##Prediction Random Forest##

from sklearn.ensemble import RandomForestClassifier

parameters = {'n_estimators' : [50,100,150,200] ,'criterion' : ['gini','entropy'], 'max_depth' : [2,5,10,12,18] }
feature_importance_data_frame = tuning_RDF("tuning_accuracy","accuracy",parameters,"Prétraitement_basique","Prediction_label_unique")

import mca

mca_ben = mca.MCA(df[["CODE","CODE_PARENT","DIAMETREARBREAUNMETRE","ESPECE","FREQUENTATIONCIBLE","GENRE_BOTA","NOTEDIAGNOSTIC","PRIORITEDERENOUVELLEMENT","SOUS_CATEGORIE","SOUS_CATEGORIE_DESC","STADEDEDEVELOPPEMENT","STADEDEVELOPPEMENTDIAG","TROTTOIR","VIGUEUR","DEFAUT"]])
mca_ind = mca.MCA(df[["CODE","CODE_PARENT","DIAMETREARBREAUNMETRE","ESPECE","FREQUENTATIONCIBLE","GENRE_BOTA","NOTEDIAGNOSTIC","PRIORITEDERENOUVELLEMENT","SOUS_CATEGORIE","SOUS_CATEGORIE_DESC","STADEDEDEVELOPPEMENT","STADEDEVELOPPEMENTDIAG","TROTTOIR","VIGUEUR","DEFAUT"]], benzecri=False)
mca_ind.expl_var(greenacre=False)
####PREDICTION MULTI LABEL####

##Premières prédictions avec un traitement simple##




##Separation Train/Test ##

y = df[["Collet","Houppier","Racine","Tronc"]]
x = df.drop(["DEFAUT","Collet","Houppier","Racine","Tronc"],axis = 1)
columns = x.columns

xtrain, xtest,ytrain, ytest = train_test_split( x, y, test_size=0.33, random_state=42)

##TreeClassifier##

tree_classifier = DecisionTreeClassifier()
tree_classifier.fit(xtrain,ytrain)

feature_importance = tree_classifier.feature_importances_
feature_importance_data_frame = pd.DataFrame(data = column_stack([columns,feature_importance]))
feature_importance_data_frame.sort_values

ypred = tree_classifier.predict(xtest)
accuracy_tree = accuracy_score(ypred,ytest)


parameters = {'criterion' : ['mse','mae','friedman_mse'], 'max_depth' : [2,5,7,10,12,15],'max_features' : [3,4,5,'auto'] }
gridsearch = GridSearchCV(estimator = tree_classifier, param_grid = parameters,scoring = 'r2',cv = 10)


##Random Forest##


from sklearn.ensemble import RandomForestClassifier
rdf =  RandomForestClassifier()  
rdf.fit(xtrain,ytrain) 

ypred2 = rdf.predict(xtest)
accuracy_rdf = accuracy_score(ypred2,ytest)

######################################################Exploration du jeu données#####################################################

###Cluster en fonction des distances###


##KMeans##
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

pos = df[["coord_x","coord_y"]]
wcss = []
for i in range(1,20):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10 ,random_state = 0)
    kmeans.fit(pos)
    wcss.append(kmeans.inertia_)
    
plt.figure()
plt.plot(range(1,20),wcss)
plt.title("elbow method")
plt.xlabel("Number of cluster")
plt.ylabel("WCSS")
plt.show()

##5 clusters##

kmeans =  KMeans(n_clusters = 5, init = 'k-means++', max_iter = 400, n_init = 10 ,random_state = 0)
y_means = kmeans.fit_predict(pos)

sls_kmeans = silhouette_score(pos,y_means)
pos_array = pos.values

plt.figure()
plt.scatter(pos_array[y_means == 0,0],pos_array[y_means == 0,1],s = 100, c = "red" , label = "cluster 0")
plt.scatter(pos_array[y_means == 1,0],pos_array[y_means == 1,1],s = 100, c = "green" , label = "cluster 1")
plt.scatter(pos_array[y_means == 2,0],pos_array[y_means == 2,1],s = 100, c = "blue" , label = "cluster 2")
plt.scatter(pos_array[y_means == 3,0],pos_array[y_means == 3,1],s = 100, c = "orange" , label = "cluster 3")
plt.scatter(pos_array[y_means == 4,0],pos_array[y_means == 4,1],s = 100, c = "black" , label = "cluster 4")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s = 300,c = "yellow",label = "centroid")
plt.title("Cluster")
plt.xlabel("coord_x")
plt.xlabel("coord_y")
plt.legend()
plt.show()

pos["cluster"] = y_means
pos["defaut"] = df["DEFAUT"]

summary_pos = pd.DataFrame(pos.groupby('cluster')['defaut'].mean())
summary_pos.plot(kind='bar')


##8 clusters##
kmeans =  KMeans(n_clusters = 8, init = 'k-means++', max_iter = 400, n_init = 10 ,random_state = 0)
y_means = kmeans.fit_predict(pos)

sls_kmeans = silhouette_score(pos,y_means)
pos_array = pos.values

plt.figure()
plt.scatter(pos_array[y_means == 0,0],pos_array[y_means == 0,1],s = 100, c = "red" , label = "cluster 0")
plt.scatter(pos_array[y_means == 1,0],pos_array[y_means == 1,1],s = 100, c = "green" , label = "cluster 1")
plt.scatter(pos_array[y_means == 2,0],pos_array[y_means == 2,1],s = 100, c = "blue" , label = "cluster 2")
plt.scatter(pos_array[y_means == 3,0],pos_array[y_means == 3,1],s = 100, c = "grey" , label = "cluster 3")
plt.scatter(pos_array[y_means == 4,0],pos_array[y_means == 4,1],s = 100, c = "black" , label = "cluster 4")
plt.scatter(pos_array[y_means == 5,0],pos_array[y_means == 5,1],s = 100, c = "orange" , label = "cluster 5")
plt.scatter(pos_array[y_means == 6,0],pos_array[y_means == 6,1],s = 100, c = "brown" , label = "cluster 6")
plt.scatter(pos_array[y_means == 7,0],pos_array[y_means == 7,1],s = 100, c = "purple" , label = "cluster 7")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s = 300,c = "yellow",label = "centroid")
plt.title("Cluster")
plt.xlabel("coord_x")
plt.xlabel("coord_y")
plt.legend()
plt.show()

pos["cluster"] = y_means
pos["defaut"] = df["DEFAUT"]

summary_pos = pd.DataFrame(pos.groupby('cluster')['defaut'].mean())
summary_pos.plot(kind='bar')




##HDBScan##
import hdbscan

##min_size : 375##

pos = df[["coord_x","coord_y"]]
hdb = hdbscan.HDBSCAN(min_cluster_size=375, gen_min_span_tree=True)
y_hdb= hdb.fit_predict(pos)
sls_hdb = silhouette_score(pos,y_hdb)

nb_clus_hdb = np.max(np.unique(y_hdb)) - np.min(np.unique(y_hdb)) + 1

plt.figure()
plt.scatter(pos_array[y_hdb == -1,0],pos_array[y_hdb == -1,1],s = 100, c = "black" , label = "outlier")
plt.scatter(pos_array[y_hdb == 0,0],pos_array[y_hdb == 0,1],s = 100, c = "red" , label = "cluster 0")
plt.scatter(pos_array[y_hdb == 1,0],pos_array[y_hdb == 1,1],s = 100, c = "green" , label = "cluster 1")
plt.scatter(pos_array[y_hdb == 2,0],pos_array[y_hdb == 2,1],s = 100, c = "blue" , label = "cluster 2")
plt.title("Cluster")
plt.xlabel("coord_x")
plt.xlabel("coord_y")
plt.legend()
plt.show()

pos["cluster"] = y_hdb
pos["defaut"] = df["DEFAUT"]

summary_pos = pd.DataFrame(pos.groupby('cluster')['defaut'].mean())
summary_pos.plot(kind='bar')

##min_size = 200##

pos = df[["coord_x","coord_y"]]
hdb = hdbscan.HDBSCAN(min_cluster_size=125, gen_min_span_tree=True)
y_hdb= hdb.fit_predict(pos)
sls_hdb = silhouette_score(pos,y_hdb)

nb_clus_hdb = np.max(np.unique(y_hdb)) - np.min(np.unique(y_hdb)) + 1

plt.figure()
plt.scatter(pos_array[y_hdb == -1,0],pos_array[y_hdb == -1,1],s = 100, c = "black" , label = "outlier")
plt.scatter(pos_array[y_hdb == 0,0],pos_array[y_hdb == 0,1],s = 100, c = "red" , label = "cluster 0")
plt.scatter(pos_array[y_hdb == 1,0],pos_array[y_hdb == 1,1],s = 100, c = "green" , label = "cluster 1")
plt.scatter(pos_array[y_hdb == 2,0],pos_array[y_hdb == 2,1],s = 100, c = "blue" , label = "cluster 2")
plt.title("Cluster")
plt.xlabel("coord_x")
plt.xlabel("coord_y")
plt.legend()
plt.show()

pos["cluster"] = y_hdb
pos["defaut"] = df["DEFAUT"]

summary_pos = pd.DataFrame(pos.groupby('cluster')['defaut'].mean())
summary_pos.plot(kind='bar')

###ANALYSE FACTORIELLE MULTIPLE###



