
Bonjour,

Ce dossier contient  l'ensemble des documents nécessaires à l'évaluation du cours de RL
Il est constitué de

- run.py qui servira à calculer la performance de mon code
- Q_function.pickle qui un matrice numpy. C'est la fonction de valeur utilisée dans FlappyPolicy.py et évaluée ici. Elle a été obtenue lors de la phase d'apprentissage décrite dans apprentissage.py
- FlappyAgent.py où est chargé la fonction de valeur qui contient la fonction FlappyPolicy appelée dans run.py
- apprentissage.py qui décrit la phase d'apprentissage


# Note pour l'évaluation

- Lorsque j'ai voulu mettre dans les conditions réelles ma fonction de valeur, les performances étaient médiocres. Je me suis rendu compte que la ligne de code dans run.py: screen = p.getscreenRGB() était la raison de ce problème. N'ayant pas besoin de cette ligne puisque je ne travaille que sur les états et pas sur l'image, serait-il possible de la commenter le temps de l'évaluation de mon travail ?

- FlappyAgent.py nécssite d'être éxecuté afin de charger la matrice Q_function nécessaire à la fonction FlappyPolicy

- Enfin, je n'ai pas compris comment faire marcher "game = FlappyBird(graphics=**"fixed"**)", donc j'ai mis "game = FlappyBird()
