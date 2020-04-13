# santorini
This repo started when I was tinkering with a simple AI for Santorini (https://en.wikipedia.org/wiki/Santorini_(game)). It is heavily inspired by the success of AlphaZero. Although the guiding example is santorini, the methods are devlopped in a way that allows a plug-n-play adaption to other games: For any game, create the corresponding abstracted graph version by overriding `gamesearch.GameGraph`. This is the main ingredient for the methods provided in `gamesearch.py` and `alphagym.py`.

## gamesearch.py
Collection of search algorithms (i.p. Monte-Carlo and alphabeta type algorithms) for two-player games. See docstring for useage examples. The central object upon which the algorithms operate is the class `GameGraph` Since for many games, it is infeasible (or impossible) to specify the full graph, we allow an 'incomplete' description: A node N may be *_open_*, that is its children are not specified, but they may be computed by some rule,which is defined in the method _add_children_. 

## santorini.py
Implements the Santorini boardgame. Base classes are `Environment` and `State`. More importantly, it abstracts the game into `SanGraph`, an overriden version of `gamesearch.GameGraph`. See docstring for useage examples.

## alphagym.py
Provides the tools generate the training examples for a neural network from selfplay. The main method is _pit_: Given the path of two tf.keras.model(s), it generates the play recors between these two neural networks which are assumed to output a prediction _pi_ for the next move and a value _v_ for each node of the the `gamesearch.GameGraph` fot the chosen game. Provides the method 'pit' in two variants: 1) distributing the network predictions over several cpu cores or 2) running it in the default single-cpu-core/gpu mode of tf.keras.

## customnet.py
Some tools do train and evaluate a simple dense Neural NEtwork with the _swish_ activation function (therfore TensorFlow 2.2 is required).

# Implementing an AlphaZero type training loop for some game
1. Encode the games structures in an instance of `gamesearch.GameGraph`, overriding the methods _nparray_of_ and _nodename_of_ if desired.
2. Impelment the Neural Network of your choice (not necessarily from `customnet.py`). Input should be compatibe with the output of the mentioned method _nparray_of_. The network should have two outputs _pi_and _v_ as explained above.
3. The loop:
    - Initialize the Neural Network with random weights and save it to disk.
    - Use a _pit_ variant and the _savedata_ function from `alphagym.py` to generate training examples from the saved model.
    - Train the model and save it to disk.
    - Use a _pit_ varriant for the saved model after training and before training to evaluate whether game performance improved.
    - Continue with selfplay with the new model, if performance increase is satisfactory, else take the old model


