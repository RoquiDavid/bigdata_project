# Project algorithme pour le big data

Ce projet a été realiser dans le cadre de l'UE "Algorithme pour le big data"

L'objectif est de prédire si un client va se voir accepter sa demande de carte de crédit. Afin de realise cela plusieurs combinaison ont été utilisée

# Feature selection

- Univariate
- PCA
- Vetor slicer

# Classification

- Gradient boosting classification
- Random forest
# Multi layer perceprton (MLP)

La combinaison retournant la meilleure precision est: PCA-MLP. 

Le fichier main contient le code permettant d'excuter la meilleure combinaison. Cependant le fichier experiment permet de voir l'execution de l'ensemble des combinaison avec leurs résultats.

# Comment executer
Comment executer:
python3 main.py
ou
python3 experiment.py

# Descriptifs des fichiers

- Les fichiers train et test sont déjà présent dans ce projet, cependant il est possible de les générer de nouveau via une fonction dans le fichier    utils.py (file_train_test_creator) puis de rename les fichier csv génerés.

- Le fichier utils.py contient toutes les fonction

- Le fichier main.py permet d'excuter le code princpal comme dit précedement (PCA-MLP)

- Le dossier data contient l'ensemble des données nécessaires

- Le fichier experiment contient l'ensemble des combinaisons avec l'affichage des résultats pour chaqune d'entre elle

