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
- le script `Train.py` contenant
le travail réalisé et détaillés ci-après.
- un dossier `models` regroupant certains des modèles générés et enregistrés.

## Guide d'utilisation

Pour entraîner les athlètes, il suffit dire à `Python` : `python Train.py`
en précisant des options si nécessaires qui sont
détaillées avec l'option `--help` (ou `-h`) par exemple: 
`python Train.py -e 1000 -x 15 -s ma_sauvegarde.pkl`.

Pour le fonctionnement avec le script `run.py`, tout est déjà en place dans le 
fichier `FlappyAgent.py` pour fonctionner avec le meilleur des modèles de type 
_feature engeneering_. Pour le modifier, il suffit de changer les variables dans
le haut du fichier pour obtenir le comportement voulu.

## Feature engineering

### Simplification
La technique utilisée pour permettre à l'algorithme de réussir l'exercice est
basée sur une simplification du vecteur d'états pour réduire l'espace d'états et
pouvoir le parcourir entièrement. Elle se base sur les hypothèses suivantes :
pour résoudre le problème de savoir s'il faut à un instant `t` battre des ailes,
ou non, il suffit de connaître :
- la distance au prochain tuyau sur un axe horizontal
- la position relative de l'oiseau par rapport au tuyau haut sur un axe vertical
- la vitesse de l'oiseau

De plus, on suppose qu'il n'est pas nécessaire d'avoir une connaissance précise 
atomique de ces valeurs, mais qu'il suffit d'en avoir une idée de l'ordre d'un 
`n`-ième de l'écran de jeu, cette valeur `n` pouvant varier entre l'axe 
horizontal et l'axe vertical, et étant modifiable. Il en va de même pour la 
vitesse et l'on appelera la quantification de cette imprécision volontaire 
l'approximation.

Ainsi, pour construire les coordonnées d'entrée de l'algorithme, on utilise les 
variables suivantes (présentes dans le vecteur d'états) :
- la distance au prochain tuyau, divisée par l'approximation horizontale donnant
ainsi une première coordonnée, que l'on note `X`
- la position verticale de l'oiseau
- la position verticale du bas du premier tuyau haut, qui soustraite à la 
position de l'oiseau et divisée par l'approximation verticale donne la position 
relative de l'oiseau par rapport au tuyau haut, notée `Y`
- enfin la vitesse que l'on divise également par l'approximation en vitesse, et 
qui sera notée finalement `V`

On a donc réduit la fonction de valeur initiale à un tableau de valeur à 3 
entrées `X`, `Y`, et `V` ayant une plage de variation raisonnablement 
explorable.

### Exploration et récompenses

Pour explorer cette espace d'états désormais simplifié, on part d'une politique
initiale consistant à toujours tomber, car on intuite que pour aller le plus
loin possible, on passe plus de temps à tomber qu'à battre des ailes. De plus, 
on ajoute à cela une part de choix aléatoire dont la proportion par rapport aux
choix appris diminue à mesure que l'apprentissage avance. Enfin, et pour pousser
l'algorithme à maximiser sa durée de vie, on biaise le système de récompenses de
la manière suivante :
- si une action amène l'oiseau à rester en vie, alors la récompense est de 1;
- si une action amène l'oiseau à passer un tuyau, alors la récompense est de 10;
- si une action amène l'oiseau à mourir, alors la récompense est de -100.

### Apprentissage
L'apprentissage est réalisé  à l'aide de TD(0) en partant de l'initialisation 
à la politique du "toujours tomber" comme énoncé précédemment. Cette méthode
suffit ici pour résoudre le problème posé, mais on peut l'améliorer en passant à
du TD(`lambda`), de manière à mettre à jour tous les états précédents plutôt que
le dernier uniquement (il faudrait alors aussi repenser le modèle de récompense)
car on se doute bien que dans la corrélation entre différents états apparaît la
notion de trajectoire.

### Résultats

Un ensemble de paramètres a été testé, dont on retrouve une partie dans les
modèles enregistrés dans le dossier `models` pour ceux qui dépassaient 15 tuyaux
sur 100 parties en moyenne. On peut conclure que cette méthode fonctionne et
permet d'aller assez loin mais souffre à mon sens d'une trop grande 
simplification du problème, car l'oiseau finit toujours par mourir sur des 
variations trop grandes de la position relative nécessaire pour passer d'un 
tuyau à un autre. Certaines fois, cette différence relative apparaît dès les 
premiers tuyaux et l'oiseau meurt tout de suite, et pour d'autres, l'oiseau 
va au delà de la centaine sans trouver pareil problème.

## Conclusion

Malgré des résultats n'allant pas au delà de 30 tuyaux en moyenne, cette version
simple d'un algorithme de Q-learning appliquée à Flappy Bird permet de résoudre 
l'exercice, et peut servir de base pour éventuellement orienter l'apprentissage 
d'un réseau de neurone. Bien qu'ayant essayé de faire du Deep-Q-Learning sur le 
vecteur d'état, je n'ai pas réussi à faire converger le réseau vers une fonction
de valeur d'une politique allant au-delà de quelques tuyaux.