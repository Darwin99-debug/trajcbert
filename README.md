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
```
