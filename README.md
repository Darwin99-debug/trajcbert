# TrajCBERT Model

## Première version du modèle TrajCBERT

### Description du modèle

Prédiction de l'avant-dernier point de la trajectoire (classification).

### Avec tout le contexte

#### Traitements des données

- Branche: **Feature/parallelization_gpu**
- Fichier: **parallelisation_gpu_deb_opti.py**

**Définir la taille du dataset avec nb_rows**

Données d'entrée: `/home/daril_kw/data/02.06.23/train_clean.json`

**Données de sortie:**
- modèle
- dataset
- tokenizer
- input_ids
- attention_masks
- labels
- full_inputs (version non encodée)

### Entrainement du modèle

Fichier de Base: **parallelisation_gpu_train_full.py**

**À modifier:**
- Nom du fichier de configuration et fichier de configuration
- Chemin d'accès des fichiers input_ids, attention_masks, labels
- Chemin d'accès du modèle
- Chemin d'accès du tokenizer

**Lancement de l'entraînement:**
`run_gpu_full.sh`


### Test du modèle 
  - Branche: **gpu_parallelization_with_torch_run**
  - Fichier: **testing_1_point_with_context.py**
## Avec seulement la destination comme information de contexte

### Description du modèle

La taille de la trajectoire est la même que celle du modèle avec contexte:

```
_CLS + 1C + POINTS + SEP + PAD_
Branche: **feature/V1_model_less_context**
```

### Traitements des données

- Branche: **feature/V1_model_less_context**
- Fichier: **parallelisation_gpu_deb_opti_only_dest.py**

**Définir la taille du dataset avec nb_rows**

Données d'entrée: `/home/daril_kw/data/02.06.23/train_clean.json`

**Données de sortie:**
- modèle
- dataset
- tokenizer
- input_ids
- attention_masks
- labels
- full_inputs (version non encodée)

### Entrainement du modèle

Fichier de Base: **parallelisation_gpu_train_full.py**

**À modifier:**
- Nom du fichier de configuration et fichier de configuration
- Chemin d'accès des fichiers input_ids, attention_masks, labels
- Chemin d'accès du modèle
- Chemin d'accès du tokenizer

**Lancement de l'entraînement:**
`run_gpu_1_2_32_20_epochs_without_context.sh` (branche gpu_parallelization_with_torch_run)


### Test du modèle 
  - Branche: **gpu_parallelization_with_torch_run**
  - Fichier: **testing_1_point_without_context.py**
# Deuxième version du modèle TrajCBERT

## Description du modèle

Le but ici est de prédire la trajectoire entière par auto-régression. Pour cela, nous devons entraîner le modèle en prédisant le point suivant de la trajectoire à n'importe quel niveau ou étape.

### Traitements des données

- Branche: **feature/generalization_2**
- Fichier de formatage: **opti_deb_gene.py** (formatage des données comme au début)
  - Output:
    - `/home/daril_kw/data/tokenizer_final`
    - `/home/daril_kw/data/data_with_time_info_ok_opti.json`
    - `/home/daril_kw/data/model_resized_embeddings`
- Fichier de formation des inputs: **opti_suite_gene_sep_dup.py** (séparation des trajectoires les plus longues selon le taux de séparation -> duplication -> séparation en catégories)
  - Inputs:
    - `/home/daril_kw/data/data_with_time_info_ok_opti.json`
    - `/home/daril_kw/data/tokenizer_final`
  - Output:
    - `/home/daril_kw/data/data_test_gene_AR_to_format.pkl`
    - `/home/daril_kw/data/AR/input_ids_v_small_AR.pt`
    - `/home/daril_kw/data/AR/attention_masks_v_small_AR.pt`
    - `/home/daril_kw/data/AR/targets_v_small_AR.pt`
    - `/home/daril_kw/data/AR/list_inputs_test_v_small_AR.pt`
    - `/home/daril_kw/data/AR/targets_dict_v_small_AR.pt`
    - `/home/daril_kw/data/AR/targets_input_v_small_AR.pt`


### Test du modèle 
#### Test auto-regressif
 - Branche: **feature/generalization_parallel**
 - Fichier: **testing_AR.py**

 ##### Pretraitement des données
  - Branche: **feature/generalization_parallel**
  - Fichier: **test_data_format_loader_AR.py**

  - Inputs: 
    - `/home/daril_kw/data/data_test_gene_AR_to_format.pkl`
    - `/home/daril_kw/data/tokenizer_final`
    - `/home/daril_kw/data/AR/targets_dict_v_small_AR.pt`
    - `/home/daril_kw/data/AR/targets_input_v_small_AR.pt`

  - Outputs:
  





#### Test avec un point

- Branche: **feature/V1_model_less_context**
- Fichier: **parallelisation_gpu_deb_opti_only_dest.py**

#### Test avec 1 ou plusieurs points
 
##### Description du fichier


Fichier de configuration: 

  - config_test_gene.json (parmètres du modèle)
  - parameter_sep_dup.json (Paramètres de formatage des données et de la version du test(Auto-régression(2) ou non(1)))



- Branche: **feature/generalization_parallel**
- Fichier: **opti_suite_sep_dup_test_format.py** 
``` 
Il est le même que celui utilisé pour les données d´entrainement mais avec la gestion des données de test en plus.
__NB__: Il faudra tester le ficher et le corriger

```


##### Fichier de lancement du test

- Fichier de test: **testing_AR.py**
  **A modifier**








