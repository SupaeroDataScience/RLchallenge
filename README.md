# RL challenge

Auteur : Thomas Cattelle

# Introduction

Ce document a pour objectif de préciser les techniques d'apprentissage utilisées pour résoudre le problème ainsi que les possibilité d'amélioration futures.

# Configuration du modèle
La plupart des hyperparamètres du modèle sont modifiables librement dans le fichier `config.py`.

# Utilisation
* Pour apprendre sans avoir de modèle pré-existant: `python train.py`
* Pour compléter l'apprentissage à partir d'un modèle pré-existant: `python retrainer.py` (le modèle existant étant le fichier pointé dans `config.MODEL_FILENAME`)
* Pour tester le modèle: `python run.py`

# Apprentissage

L'apprentissage du jeu est basé **uniquement sur les pixels** de l'écran (variable `screen`). Le vecteur d'état n'est jamais utilisé, ni pendant l'apprentissage, ni pendant la phase de test.

## DQN
Le modèle est entrainé par le biais d'un DQN de la structure suivante:

###Couches
1. Convolution 2D, 32 filters, kernel (8,8), strides 4, suivie d'une couche d'activation ReLu
2. Convolution 2D, 64 filters, kernel (4,4), strides 2, suivie d'une couche d'activation ReLu
3. Convolution 2D, 64 filters, kernel (3,3), strides 1, suivie d'une couche d'activation ReLu
4. Couche de 512 neurones complètementement connectés, suivie d'une couche d'activation ReLu
5. Couche de sortie de 2 neurones (2 actions) complètement connectée, couche d'activation linéaire

###Initialisation
Les trois couches convolutives sont initialisées par un tirage selon la loi normale avec une moyenne à 0 et un écart type de 0.1. Les tirages qui se situent à plus de deux écarts-types de la moyenne sont tirés à nouveau (`keras.initializers.TruncatedNormal`)

###Optimiseur
RMSProp avec un taux d'apprentissage initial de 1e-6 et un decay de 0.9.

###Pré-traitement
L'écran (288*512 par défaut) est converti en niveaux de gris, rogné de (0,0) jusqu'à (405,288) (afin d'éliminer les textures statiques du sol), puis enfin réduit et redimensionné en (84,84). Ce pré-traitement est réalisée à l'aide de la bibliothèque Pillow.

###Entrée
On fournit au DQN une entrée de la taille (`batch_size`, `history_length`, 84, 84) où `batch_size` correspond à la taille d'une minibatch d'entrainement (32 par défaut) et `history_length` correspond au nombre d'écrans (*frame*) que l'on conserve dans un état (4 par défaut).

###Paramètres d'entraînement
Par défaut, le modèle génère des échantillons pour la mémoire d'Experience Replay pendant 3000 itérations puis apprend pendant 597000 itérations (600000 itérations en tout).

###Reward shaping
Les récompenses sont modifiées dans la façon suivante:
* -1.0 **exactement** si l'oiseau meurt (la récompense par défaut n'est pas fixe dans le simulateur de base)
* +0.1 si l'oiseau survit pendant une frame
* +1.0 si l'oiseau passe un pipe

# Résultats
Les résultats sont peu encourageants à l'heure actuelle et le modèle a toujours du **mal à généraliser** afin d'obtenir la politique optimale (qui consiste généralement à rester vers le bas de l'écran et à voler juste avant un tuyau). Le modèle témoigne toutefois d'un véritable apprentissage en cela qu'il essaie souvent de "viser" l'espace du tuyau.

En particulier, l'oiseau se cogne de façon récurrente juste après avoir passé un pipe. Une solution envisageable sera l'implémentation d'une méthode de **frame skipping** qui consiste à apprendre et à prendre une décision à l'aide du DQN tous les `n` frames plutôt qu'à chaque frame. Cette piste n'a pas pu être explorée à temps en raison du long temps d'apprentissage du modèle.

Enfin, certaines trajectoires aberrantes suivies par l'oiseau (e.g. monter directement jusqu'en haut de l'écran en tout début de partie) suggère l'existence de boucles de rétroaction néfastes dans le modèle lors de l'entrainement. Une solution possible serait alors de dédoubler le modèle d'apprentissage en deux DQN:
* Un DQN primaire qui apprendrait à chaque itération comme c'est le cas actuellement
* Un DQN secondaire mis à jour avec les poids du DQN primaire toutes les `n` itérations et qui est chargé d'effectuer les prédictions d'actions. Ce DQN secondaire est alors celui utilisé en phase de test.

Cette architecture bicéphale permettrait possiblement de stabiliser l'apprentissage.