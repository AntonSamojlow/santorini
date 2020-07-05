# Intro
This repo started when I was tinkering with a simple AI for [Santorini](https://en.wikipedia.org/wiki/Santorini_(game)). The approach here is heavily inspired by the success of [AlphaZero](https://en.wikipedia.org/wiki/AlphaZero): A reinforcement learning loop, where the data for training is generated via selfplay. This project tries also to answer whether a sufficiently large set of self play data for a good training result can be generated when only using one desktop computer. For this, the program uses several parallel processes (in particular several selfplay processes with their own set of worker threads) to try maximizing the limited set of computational ressources.

There are two main ingredients for the program, the first is the abstraction of the game into a graph: the class `gamegraph.GameGraph`. The second is a neural network (only tensorflow.keras is supported at the moment) which is compatible with the graph. Compatible means that the output of `gamegraph.GameGraph.numpify(vertex)` is a valid input for the neural network.   


### Requirements
- python 3.8.3
- tensorflow 2.2.0
- numpy 1.18.4
- matplotlib 3.2.1 (only for saving training plots)

### Examplescript 
See `script-trail.py` together with the set of initial models in the folder '/initialmodels' for an example. It uses the mentioned boardgame 'Santorini'.

Santorini is only a guiding example. The code is designed to allow a plug-n-play adaption to other games: For any game, create the corresponding graph version by implementing the abstract class `gamegraph.GameGraph` (like `santorini.SanGraph` does for example ). For more details on how to use the code to train on another game see a later section below.


# Training loop and configuration
### Flow overview
The GameGym session will start one Trainer, one Predictor and **selfplayprocesses**-many Selfplayer processes. Each Sefplayer itself will continously play against itself by selecting the move from the children (possible moves/next gamestates) of the current vertex (gamestate) with a probability given by the visit count statistics of **searchcount**-many MCTS-type searches. Each game result is a sequence of states _x_ annotated with a value _y_val_ = +1/-1 (encoding on whether the active player of that state _x_ won) and the MCTS-result of move probabilities _y_pi_. These pairs (_x_, _y_val_, _y_pi_) are the data used for the training of the neural network. They are peridoically dumped to disk and read by the Trainer. After each training, the Predictor is signalled to update its model.

### Monte Carlo Tree Search
The MCTS-type search is performed on the current vertex in parallel by **searchthreadcount**-many workers. Each workers walks down the graph by a LCB rule (UCB algorithm for selecting the vertex with _minimal_ value) until reaching an open or terminal vertex (the select phase). On an open vertex, the worker will expand and flag it as being under expansion. In case of a non-terminal vertex, probabilities and value estimate for the statistics table are initialized from a neural network prediction (forward pass). This replaces the usual simulation step of a classic MCTS. All workers of all Selfplayers query the Predictor process for this, using the speedup gain of a batch prediction. The flag is removed when the predict request has been answered. Using a lock for the shared statistics table, the worker updates the table itself. Compared to an centrallized update approach, this avoids that workers start a new searchrun before their own update has been processed.

All workers that have already selected a vertex which is currently under expansion by another worker are forced to wait until the expansion and prediction are complete. **Virtualloss** is therefore applied to encourage the selection of other vertices.

### Training and iteration count
The train process uses the pre-defined optimzier of the compiled tf.keras model. After each training run (see parameter **epochs**), the model is saved and its iteration count increased. The selfplay data is marked by the iteration count of the Predictor network used. Before training, the Trainer will load all samples from disk whose iteration count mark, compared to the current iteration count of the network, does not exceed **max_sampleage**. For all theses samples, maximally **max_samplecount** are selected. The Trainer waits until at least **min_samplecount** samples are available for training (after the **validation_split**).

The Training procedure can and should be improved at many places. For example by an early stopping criterion or allowing the optimizer and regularization to change while keeping the current weights. Also an eavaluation of different models performance by pitting them against each other would is needed to track progress. This is _planned future work..._


# Details on project files
Some details on the python files within this project:

## santorini.py
Implements the two-player boardgame [Santorini](https://en.wikipedia.org/wiki/Santorini_(game)). The class `SanGraph` is an implementation of the abtract class `gamesearch.GameGraph` and rqeuires information about the boardsize (3 to 5) and the unitcount per player (1 or 2). This is passed by the class `Environment`. Note that in the case of a 3x3 board and 1 unit per player, one can use an alphabeta search at depth 12 to compute the optimal policy. This can serve as a benchmark. 

## gamegym.py
A GameGym instance represents a session of the reinforcement learning (RL) loop and is tied to a folder where the configuration, state, progress and logs are saved. The resume method starts/continues the RL loop. It accepts the keyboard input 'exit' from the console to shutdown the loop. While running, several processes are started and managed:
- Selfplayer(s): Generate the training material via selfplay. Each selfplay uses
several worker threads to execute a parallelized MCTS search (where simulation is replaced 
by neural network predictions)
- Predictor: Answers predictions for all Selfplayers
- Trainer: Continiusly trains the network on the most recent selfplay data

The code for these processes ca be found in '/processes'. Informational log output of the main process (gamegym) is written to console. Detailed log of the main and sub processes is written to disk.

## gymdata.py
Provides data structures and tools for those. In particular, configuration options for the RL and their default values are defined here.

## gamegraph.py
Represents an abstraction of the underlying decision graph of a _symmetric_ two-player game. Each vertex in the graph is a state of the game, seen from the active players perspective. The roots are the possible starting positions of the game and the children of each vertex are the possible game states that can be reached by a valid action (move) of the active player. The convention for the score of a state used here is: it lies between -1 and +1 and is -1/+1 iff the state is terminal with the active player having lost/won. This convention sets the basis for the value-estimate which the RL loop uses to approximate the optimal policy. 

It is unfeasible to compute the full graph structure of most games. Hence the class GameGramp allows a vertex to be _open_, meaning its children have not been explored yet (need to call `GameGraph.expand_at(vertex)`). It can can in particular not be checked if an open vertex is terminal. And a vertex being terminal implies it is not open. 

## customnetwork.py
Some tools do train and evaluate a simple dense Neural Network with the _swish_ activation function (therefore TensorFlow 2.2 is required). Can and maybe should be replace by a different neural network design (_planned future work..._).

# Implementing the training loop for your own game
1. Encode the games structures by implementing the abstract class `gamesearch.GameGraph` (inherit and overwrite at least all abstract methods).
2. Implement the Neural Network of your choice via `tensorflow.keras` and save it to disk. The network input should be compatibe with the output of `[your_GameGraph].numpify(vertex)`.
3. Configure the GameGym session and pass the networks folder as initialmodelpath. See `script-trialrun.py` for an example of configuration options.

