# RL challenge

RL class' challenge files

# Your job

Your job is to:
<ol>
<li> fork the project at ... on your own github (yes, you'll need one).
<li> move the only file there ('run.py') under a directory "YourLastName".
<li> create 'FlappyPolicy.py' in order to implement the function `FlappyPolicy(state,screen)` used below. You're free to add as many extra files as you need. However, you're not allowed to change 'run.py'.

<li> add any useful material (comments, text files, analysis, etc.)
<li> Make a pull request on the original repository when you're done.
</ol>

`FlappyPolicy(state,screen)` takes both the game state and the screen as input. It gives you the choice of what you base your policy on:
<ul>
<li> If you use the state variables vector and perform some handcrafted feature engineering, you're playing in the "easy" league. If your agent reaches an average score of 15, you're sure to have a grade of at least 10/20 (possibly more if you implement smart stuff and/or provide a smart discussion).
<li> If you use the state variables vector without altering it (no feature engineering), you're playing in the "good job" league. If your agent reaches an average score of 15, you're sure to have at least 15/20 (possibly more if you implement smart stuff and/or provide a smart discussion).
<li> If your agent uses only the raw pixels from the image, you're playing in the "Deepmind" league. If your agent reaches an average score of 15, you're sure to have at the maximum grade (plus possible additional benefits).
<ul>
