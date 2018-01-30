# RL challenge

Your challenge is to learn to play [Flappy Bird](https://en.wikipedia.org/wiki/Flappy_Bird)!

Flappybird is a side-scrolling game where the agent must successfully nagivate through gaps between pipes. Only two actions in this game: at each time step, either you click and the bird flaps, or you don't click and gravity plays its role.

There are three levels of difficulty in this challenge:
- Learn an optimal policy with hand-crafted features
- Learn an optimal policy with raw variables
- Learn an optimal policy from pixels.

# Your job

Your job is to:
<ol>
<li> fork the project at [https://github.com/SupaeroDataScience/RLchallenge](https://github.com/SupaeroDataScience/RLchallenge) on your own github (yes, you'll need one).
<li> rename the "RandomBird" folder into "YourLastName".
<li> modify 'FlappyPolicy.py' in order to implement the function `FlappyPolicy(state,screen)` used below. You're free to add as many extra files as you need. However, you're not allowed to change 'run.py'.
<li> you are encouraged, however, to copy-paste the contents of 'run.py' as a basis for your learning algorithm.
<li> add any useful material (comments, text files, analysis, etc.)
<li> make a pull request on the original repository <i>when you're done</i> (please don't make a pull request before you think your work is ready to be merged on the original repository).
</ol>

**All the files you create must be placed inside the directory "YourLastName".**

`FlappyPolicy(state,screen)` takes both the game state and the screen as input. It gives you the choice of what you base your policy on:
<ul>
<li> If you use the state variables vector and perform some handcrafted feature engineering, you're playing in the "easy" league. If your agent reaches an average score of 15, you're sure to have a grade of at least 10/20 (possibly more if you implement smart stuff and/or provide a smart discussion).
<li> If you use the state variables vector without altering it (no feature engineering), you're playing in the "good job" league. If your agent reaches an average score of 15, you're sure to have at least 15/20 (possibly more if you implement smart stuff and/or provide a smart discussion).
<li> If your agent uses only the raw pixels from the image, you're playing in the "Deepmind" league. If your agent reaches an average score of 15, you're sure to have at the maximum grade (plus possible additional benefits).
</ul>

Recall that the evaluation will start by running 'run.py' on our side, so 'FlappyPolicy' should call an already trained policy, otherwise we will be evaluating your agent during learning, which is not the goal. Of course, we will check your learning code and we will greatly appreciate insightful comments and additional material like (documentation, discussion, comparisons, perspectives, state-of-the-art...).

# Installation

You will need to install a few things to get started.
First, you will need PyGame.

```
pip install pygame
```

And you will need [PLE (PyGame Learning Environment)](https://github.com/ntasfi/PyGame-Learning-Environment) which is already present in this repository (the above link is only given for your information). To install it:
```
cd PyGame-Learning-Environment/
pip install -e .
```
Note that this version of FlappyBird in PLE has been slightly changed to make the challenge a bit easier: the background is turned to plain black, the bird and pipe colors are constant (red and green respectively).
