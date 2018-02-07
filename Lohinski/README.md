# SDD - Challenge RL : Fly Flappy Bird, fly !
**Auteur**: Nikola Lohinski

## Introduction
Ce document a pour but de rendre plus claire l'organisation du repo et de mettre
en évidence la logique du travail réalisé.

Le répertoire est constitué de :
- ce document **README.md**
- le fichier **run.py** inchangé
- un fichier **requirements.txt** contenant l'ensemble des dépendances du
projet. Il n'est pas indispensable mais est une bonne pratique et permet de 
tout installer facilement en faisant **pip install -r requirements.txt**.
- le code source **FlappyAgent.py** qui ne fait pas grand chose à part appeler
d'autres scripts.
- les codes **StateEngineeringAthlete.py** et **DeepStateAthlete.py** contenant
le travail réalisé et détaillés ci-après.
- un dossier **models** regroupant certains modèles générés et enregristrés.

## Guide d'utilisation
Pour entraîner nos athlètes, il suffit de lancer directement le script de 
l'athlète concerné et en précisant des options si nécessaires qui sont
détaillées lorsqu'on ajoute **-h** ou **--help** (ou n'importe
quelle action non répertoriée) par exemple: 
**python StateEngineeringAthlete.py -h**.

Pour le fonctionnement avec le script **run.py**, tout est déjà en place dans le 
fichier **FlappyAgent.py** pour fonctionner avec le modèle de type _feature 
engeneering_. Pour le modifier, il suffit de changer les variable dans le haut 
du fichier pour obtenir le comportement voulu.

## Feature engineering

## Deep Q learning