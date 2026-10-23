# CHALLENGE-DATA-ENS-CREDIT-AGRICOLE

# 📑 Challenge Data ENS — AssurPrime (Crédit Agricole Assurances)

Projet de prédiction de la **prime pure liée au risque incendie** pour des contrats multirisques agricoles dans le cadre du Challenge Data ENS.

**Auteures :** Héléna Cros & Éloïse Sallet

---

## 🎯 Objectif & Défi

* **Variable cible (`CHARGE`) :** $\text{Fréquence} \times \text{Coût moyen} \times \text{Exposition}$
* **Données :** 383 610 contrats et 374 variables (bâtiments, cultures, météo, capitaux...)
* **Défi majeur :** Jeu de données extrêmement déséquilibré avec seulement **2 352 sinistres réels** ($>0$)

---

## 🏆 Résultats & Démarche

Plusieurs modèles de Machine Learning et Deep Learning ont été évalués avec la métrique **RMSE** :

| Stratégie | Modèles & Techniques | Score RMSE |
| :--- | :--- | :---: |
| **LightGBM (2 modèles)** | Fréquence (Poisson) + Coût Moyen (Gamma) | **5 600** |
| **LightGBM (Tweedie)** | Modèle unifié Tweedie ($p=1.5$) | **5 600** |
| **Réseau de Neurones (ANN)** | Dense ANN avec perte Tweedie (Log-link) | ~ 5 602 |
| **Data Augmentation** | Génération de 5 000 sinistres synthétiques via **CTGAN** | **5 600** |
| **Pipeline Final (Meilleur)** | **LightGBM Tweedie + Validation croisée 5-Fold + CTGAN** | **🏆 5 599** |

---

## ⚖️ Équité (Disparate Impact)

Analyse du **Disparate Impact (DI)** sur la variable `ACTIVIT2` (type d'activité) :
* Les écarts observés (ex. `ACT8` à $0.757$) s'expliquent par un **risque incendie réellement plus élevé** (stockage de foin, céréales...) et non par un biais du modèle.

---

## 🛠️ Stack Technique

* **Langage :** Python 3.13
* **Librairies :** `lightgbm`, `tensorflow`, `ctgan`, `scikit-learn`, `pandas`

---

## 📂 Fichiers du dépôt

* `notebook_cros_sallet.ipynb` : Code source complet et modélisations.
* `rapport_ter_cros_sallet.pdf` : Rapport d'analyse détaillé.
* Les fichiers de données (X_train, Y_train et X_test) sont trop volumineux pour github mais sont disponibles ici : https://challengedata.ens.fr/participants/challenges/161/
