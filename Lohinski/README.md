# SDD - Challenge RL : Fly Flappy Bird, fly !
**Auteur**: Nikola Lohinski

## Introduction
Ce document a pour but de rendre plus claire l'organisation du code et de mettre
en évidence la logique du travail réalisé.

Le répertoire est constitué de :
- ce document `README.md`
- le fichier `run.py` inchangé
- un fichier `requirements.txt` contenant l'ensemble des dépendances du
projet. Il n'est pas indispensable mais est une bonne pratique et permet de 
tout installer facilement en faisant `pip install -r requirements.txt`.
- le code source `FlappyAgent.py` qui ne fait pas grand chose à part appeler
d'autres scripts.
- les codes `Train.py` et `DeepTrain.py` contenant
le travail réalisé et détaillés ci-après.
- un dossier `models` regroupant certains modèles générés et enregristrés.

## Guide d'utilisation
Pour entraîner les athlètes, il suffit de lancer directement le script de 
l'athlète concerné en précisant des options si nécessaires qui sont
détaillées avec l'option `--help` (ou `-h`) par exemple: 
`python Train.py -h`.

Pour le fonctionnement avec le script `run.py`, tout est déjà en place dans le 
fichier `FlappyAgent.py` pour fonctionner avec le modèle de type _feature 
engeneering_. Pour le modifier, il suffit de changer les variables dans le haut 
du fichier pour obtenir le comportement voulu.

## Feature engineering

Pour entraîner l'athlète de type _feature engineering_ contenu dans `Train.py`,
l'espace d'états a été simplifié de la façon suivante :
- seules 4 composantes du vecteur d'état ont été utilisées:
    - la hauteur de l'oiseau par rapport au sol `player_y`
    - la hauteur du prochain tuyau haut `next_pipe_top_y`
    - la distance de l'oiseau au prochain tuyau `next_pipe_dist_to_player`
    - la vitesse de l'oiseau `player_vel`
- ces 4 composantes sont combinées dans la fonction `state2coord` pour se
ramener à 3 variables (une position réduite relative en X, une position réduite relative en Y, 
et une vitesse réduite) de domaine plus restreint que l'espace d'états initial
- selon les modèles (c'est à dire selon les valeurs de `x_reduce`, `y_reduce`
et `v_reduce` qui sont les facteurs de simplifications de l'axe X, l'axe Y et 
l'axe des vitesses respectivement), l'espace d'états est plus ou moins gros
et cela prend plus ou moins de temps pour l'explorer et converger.
- une trop grande simplification ne permet pas de trouver une stratégie optimale
et un espace d'état de grand empêche de conclure sur la convergence rapidement

On constate que les valeurs de `alpha = 0.65` et `gamma = 0.85` fonctionnent bien
quel que soit le modèle de simplification.
De plus, en notant `(X,Y,V)` un modèle de simplification, on a :
- `(15, 15, 2)` qui converge en moins de 10000 itérations vers une solution à 
moyenne supérieur à 15 tuyaux passés mais finit par désapprendre si on le laisse
tourner trop longtemps. 
- `(10, 10, 1)` qui converge plus lentement mais est beaucoup plus stable si l'on
continue l'apprentissage et permet d'atteindre des score moyens de l'ordre de la 
trentaine.

Un tel modèle permet de résoudre le jeu et de faire de bons scores, et peut 
notamment servir à entraîner un algorithme de deep-Q-learning dans sa phase
exploratoire au lieu d'explorer aléatoirement.

## Deep Q learning