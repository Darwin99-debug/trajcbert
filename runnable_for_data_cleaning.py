# %% [markdown]
# # Imports

# %%
!pip install pandas
import numpy as np
import pandas as pd
from numpy import genfromtxt
import matplotlib.pyplot as plt
import ast
import json

# %%
import gc 
gc.collect()

# %%
import statistics as stat

# %% [markdown]
# # Chargement des données

# %% [markdown]
# Ici pour l'instant on teste sur une version raccourcie du jeu de données sinon c'est trop long. 

# %%
# Opening JSON file small version
#with open('train_porto_tokenized_small_1st_partV2.json', 'r') as openfile:
# Opening JSON file full version
with open('train_porto_tokenized.json', 'r') as openfile:

 
    # Reading from json file
    json_loaded = json.load(openfile)
 
# print(json_loaded)
print(type(json_loaded))

# %% [markdown]
# On met en dataframe les données chargées.

# %%
data_loaded = pd.DataFrame(data=json_loaded)

# %% [markdown]
# On vérifie la tête de nos données et leur type :

# %%
data_loaded.info()

# %%
#type(data_loaded['POLYLINE']), data_loaded['POLYLINE']

# %%
#type(data_loaded['POLYLINE'][0]), data_loaded['POLYLINE'][0] 

# %%
#type(data_loaded['POLYLINE'][0][0] ), data_loaded['POLYLINE'][0][0]  

# %%
#type(data_loaded['POLYLINE'][0][0][0] ), data_loaded['POLYLINE'][0][0][0]  

# %%
#type(data_loaded['Tokenization'][0][0] ), data_loaded['Tokenization'][0][0] 

# %%
data_loaded.head()

# %% [markdown]
# __Conclusion :__
# 
# On a bien les données dans le format voulu.

# %% [markdown]
# # Séparation des trajectoires en début (input)/point(s) à prédire/destination

# %% [markdown]
# ## Vérification de la tokenization
# 
# Le but est de vérifier la taille des listes de tokens avec la liste de points géographique correspondant.

# %%
def count_points(traj):
    return len(traj)

# %% [markdown]
# On vérifie le fonctionnement sur une ligne :

# %%
count_points(data_loaded['POLYLINE'][0])

# %%
data_loaded['POLYLINE'][0]

# %% [markdown]
# On définie une fonction ajoutant une colonne rescensant le nombre de points dans la trajectoire non tokenizée et une pour la trajectoire tokenizée.

# %%
def verif(data_load):
    data_csv=data_load
        
    data_csv['Nb_points'] = data_loaded['POLYLINE'].apply(lambda x: count_points(x))
    data_csv['Nb_points_token'] = data_loaded['Tokenization'].apply(lambda x: count_points(x))
        

    return data_csv     

# %% [markdown]
# Enfin, on définie une fonction qui compare les deux longueurs pour chaque trajectoire, et bon l'appelle.

# %%
def meme_longueur(data_load) : #retourne vrai si la liste de tokens et celle de points geographiques sont de même taille
    dataframe=verif(data_load)
    data_liste = (dataframe['Nb_points']==dataframe['Nb_points_token'])
    listebool=data_liste.to_numpy()
    return (not(False in listebool))

# %%
meme_longueur(data_loaded)

# %% [markdown]
# __Conclusion :__
# 
# On a bien les bonnes tailles de listes des points tokenisés.
# On remarque aussi que les trajectoires ne sont pas toutes de même longueur.
# 
# On peut donc ne garder qu'une des deux colonnes et procéder à la suite.

# %%
data_loaded['Nb_points']=data_loaded['POLYLINE'].apply(lambda x: count_points(x))

# %%
data_loaded.head()

# %%
#data_loaded.drop('Nb_points_token',axis=1)

# %%
data_loaded.info()

# %% [markdown]
# ## Troncature

# %% [markdown]
# Le problème que nous rencontrons est que BERT a pour entrée maximale 512 tokens. Or certaines trajectoires dépasse ce seuil. Afin de pouvoir utiliser BERT dans le cas de longues trajectoires, plusieurs s'offrent à nous. Une d'elle consiste à ne considérer que la fin de la trajectoire étudiée si elle est trop longue. C'est celle que nous mettons en oeuvre ici.
# 
# Nous avons besoin de connaitre la taille que prendra le contexte à concaténer pour rentrer dans la limite des 512 tokens après concaténation. Nous la fixons avec la variable "limite_entrée" à 400 pour le moment.
# 
# On commence par une étape d'étude de nos données : on regarde la proportion de trajectoire trop longues, les longueurs moyennes, ... 

# %%
## On commence par faire une copie de data_loaded au cas où on ferait des modifications non voulues de data_loaded et de perdre la bonne version
df=data_loaded.copy()

# %%
## On se donne une idée de la longueur max des trajectoires considérées

def recup_max(dataset, col):
    maximum=0
    for i in range(len(dataset[col])):
        if maximum<dataset[col][i]:
            maximum=dataset[col][i]
            position=i
    return [maximum,position]

#On peut rérer le max directement avec max(dataf['Nb_points']) mais cela donne pas la ligne.

# %%
maximum,position=recup_max(df, 'Nb_points')
maximum,position

# %%
"""
#Fonction pour avoir une idée du nombre/de la proportion de trajectoires trop longues

max_bert = 512

def compter_cb_trop_long(dataset,col):
    liste=[]
    for i in range(len(dataset[col])):
        if dataset[col][i]>max_bert:
            liste.append(i)
    return [len(liste),liste]

compteur,lis=compter_cb_trop_long(df,'Nb_points')
print(compteur,compteur/len(df.index))
    

"""


# %%
longueur_moyenne = df['Nb_points'].mean()
longueur_moyenne

# %%
stat.median(df['Nb_points'])

# %%
stat.mode(df['Nb_points'])

# %%
stat.pstdev(df['Nb_points'])

# %%
stat.pvariance(df['Nb_points'])

# %%

limite_entree=400 #trajectoire maximale qu'on fixe (nb de jetons)


def troncature(datasetav,datasetap,limite_entree): # modifie le datasetap sans toucher au datasetav
#Retourne le nb de trajectoire modifiées

    nb=0
    for i in range(len(datasetav['Nb_points'])): # on parcourt les lignes de la colonne 'Nb_points'
        longueur=datasetav['Nb_points'][i]
        if longueur>limite_entree: #si trop long
            nb+=1 
            
            
            #Si on veut garder le début de la trajectoire :
            # datasetap['POLYLINE'][i]=datasetav['POLYLINE'][i][:limite_entree] 
            # datasetap['Tokenization'][i]=datasetav['Tokenization'][i][:limite_entree]    
            
            #Si on veut garder la fin de la trajectoire :
            indice_deb= longueur-limite_entree
            datasetap['POLYLINE'][i]=datasetav['POLYLINE'][i][indice_deb:] 
            datasetap['Tokenization'][i]=datasetav['Tokenization'][i][indice_deb:]    
            
            
    return nb

# %%
#On garde des copies
data_truncated=df.copy()

# %%
#On effectue la troncature
troncature(df,data_truncated,limite_entree)

# %%
data_truncated.head()

# %%
#on maj le nb de points dans la trajectoire

data_truncated['Nb_points'] = data_truncated['POLYLINE'].apply(lambda x: count_points(x))

# %%
max(data_truncated['Nb_points'])

# %% [markdown]
# On a obtenu une version avec les trajectoires tronquées pour ne pas avoir de trajectoires trop longues.

# %% [markdown]
# ## Détection de trajectoires trop courtes
# Le but est d'enlever des potentielles trajectoires de moins de 3 points.

# %%
data_truncated_copy=data_truncated.copy()

# %%
def liste_traj_trop_courtes(dataframe_a) :
    index_to_remove =[]
    for i in range(len(dataframe_a['POLYLINE'])):
        if dataframe_a['Nb_points'][i]<3:
            index_to_remove.append(i)
    return index_to_remove
    

# %%
liste_index_a_enlever=liste_traj_trop_courtes(data_loaded)

# %%
len(liste_index_a_enlever)

# %%
liste_traj_trop_courtes(data_loaded) == liste_traj_trop_courtes(data_truncated)

# %%
len(liste_index_a_enlever)/len(data_truncated.index), len(liste_index_a_enlever)

# %%
liste_index_a_enlever[0]

# %%
data_loaded.iloc[54].name

# %%
type(liste_index_a_enlever[0])

# %%

def to_string_with_cote (liste):
    for i in range(len(liste)):
        liste[i] = str(liste[i])

    return liste

# %%
to_string_with_cote ([1,2,3])

# %%
def remove_traj_too_short(dataframe):
    list_traj_to_remove = liste_traj_trop_courtes(dataframe)
    list_traj_to_remove=to_string_with_cote(list_traj_to_remove)
    dataframe_new=dataframe.drop(index=list_traj_to_remove)
    
    return dataframe_new
    
        

# %%
data_clean=remove_traj_too_short(data_loaded)

# %%
data_clean_truncated=remove_traj_too_short(data_truncated)

# %%
len(data_clean.index)==len(data_loaded.index)-len(liste_index_a_enlever)

# %%
data_loaded.index

# %%
data_clean.iloc[54]

# %%
# Serializing json
json_object = json.dumps(data_clean.to_dict())

# Writing to sample.json
with open("train_clean.json", "w") as outfile:
    outfile.write(json_object)

# %%
# Serializing json
json_object = json.dumps(data_clean_truncated.to_dict())

# Writing to sample.json
with open("train_clean_truncated.json", "w") as outfile:
    outfile.write(json_object)

# %% [markdown]
# ## Définition des fonctions de séparation

# %% [markdown]
# Pour le moment, on ne veut prédire qu'un seul point.

# %%
def padding_et_sep(traj):
    destination = traj[-1]
    a_predire= traj[-2]
    deb=traj[:-2] #on enleve les deux derniers /!\ pas de = car pop modifie la liste et renvoie l'élément enlevé
    while len(deb)<limite_entree-2:
        deb.append('[PAD]')
    return (deb, a_predire, destination )
    


# %%
traj_input, traj_target, traj_dest = separation(dataf['POLYLINE'][0])

# %%
len(traj_input)

# %%
traj_input

# %%
traj_target , traj_dest

# %%
df_to_load['traj_input'],df_to_load['traj_target'],df_to_load['traj_dest']=dataf['POLYLINE'].apply(lambda x: separation(x))

# %%



