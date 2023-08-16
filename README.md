# Stock Price Prediction Using Recurrent Neural Networks (RNN)

Stock market prediction and analysis are some of the most difficult jobs to complete. There are numerous causes for this, including market volatility and a variety of other dependent and independent variables that influence the value of a certain stock in the market. These variables make it extremely difficult for any stock market expert to anticipate the rise and fall of the market with great precision.

However, with the introduction of Machine Learning and its strong algorithms, the most recent market research and Stock Market Prediction advancements have begun to include such approaches in analyzing stock market data.

### Stock Price Prediction [Code](https://github.com/anupam215769/Stock-Price-Prediction-RNN-DL/blob/main/rnn.ipynb) OR <a href="https://colab.research.google.com/github/anupam215769/Stock-Price-Prediction-RNN-DL/blob/main/rnn.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

> Don't forget to add Required Data files in colab. Otherwise it won't work.


## Recurrent Neural Networks

Recurrent neural networks, or RNNs, are a type of artificial neural network that add additional weights to the network to create cycles in the network graph in an effort to maintain an internal state.

The promise of adding state to neural networks is that they will be able to explicitly learn and exploit context in sequence prediction problems, such as problems with an order or temporal component.

![rnn](https://i.imgur.com/exNBhlz.png)

- **In 3D**

![rnn](https://i.imgur.com/35RC36q.png)

- **In 2D**

![rnn](https://i.imgur.com/Ak3veCd.png)

## Vanishing and Exploding Gradient Problem

- **Vanishing Gradient Problem**

Vanishing Gradient occurs when the derivative or slope will get smaller and smaller as we go backward with every layer during backpropagation.

When weights update is very small or exponential small, the training time takes too much longer, and in the worst case, this may completely stop the neural network training.

A vanishing Gradient problem occurs with the sigmoid and tanh activation function because the derivatives of the sigmoid and tanh activation functions are between 0 to 0.25 and 0–1. Therefore, the updated weight values are small, and the new weight values are very similar to the old weight values. This leads to Vanishing Gradient problem. We can avoid this problem using the ReLU activation function because the gradient is 0 for negatives and zero input, and 1 for positive input.

![rnn](https://i.imgur.com/Ak3veCd.png)

- **Exploding Gradient Problem**

Exploding gradient occurs when the derivatives or slope will get larger and larger as we go backward with every layer during backpropagation. This situation is the exact opposite of the vanishing gradients.

This problem happens because of weights, not because of the activation function. Due to high weight values, the derivatives will also higher so that the new weight varies a lot to the older weight, and the gradient will never converge. So it may result in oscillating around minima and never come to a global minima point.

![rnn](https://i.imgur.com/QeINet9.png)

Vanishing gradient problem is far more threatening as compared to the exploding gradient problem, where the gradients become very very large due to a single or multiple gradient values becoming very high.

The reason why Vanishing gradient problem is more concerning is that an exploding gradient problem can be easily solved by clipping the gradients at a predefined threshold value. Fortunately there are ways to handle vanishing gradient problem as well. There are architectures like the LSTM(Long Short term memory) and the GRU(Gated Recurrent Units) which can be used to deal with the vanishing gradient problem.

![rnn](https://i.imgur.com/ZwDRDtP.png)


## Long Short-Term Memory

With conventional Back-Propagation Through Time (BPTT) or Real Time Recurrent Learning (RTTL), error signals flowing backward in time tend to either explode or vanish.

The temporal evolution of the back-propagated error exponentially depends on the size of the weights. Weight explosion may lead to oscillating weights, while in vanishing causes learning to bridge long time lags and takes a prohibitive amount of time, or does not work at all.

- LSTM is a novel recurrent network architecture training with an appropriate gradient-based learning algorithm.
- LSTM is designed to overcome error back-flow problems. It can learn to bridge time intervals in excess of 1000 steps.
- This true in presence of noisy, incompressible input sequences, without loss of short time lag capabilities.
 
Error back-flow problems are overcome by an efficient, gradient-based algorithm for an architecture enforcing constant (thus neither exploding nor vanishing) error flow through internal states of special units. These units reduce the effects of the “Input Weight Conflict” and the “Output Weight Conflict.”

![rnn](https://i.imgur.com/ZwDRDtP.png)

- **The Input Weight Conflict:** Provided the input is non-zero, the same incoming weight has to be used for both storing certain inputs and ignoring others, then will often       receive conflicting weight update signals.

  These signals will attempt to make the weight participate in storing the input and protecting the input. This conflict makes learning difficult and calls for a more context-     sensitive mechanism for controlling “write operations” through input weights.

- **The Output Weight Conflict:** As long as the output of a unit is non-zero, the weight on the output connection from this unit will attract conflicting weight update signals    generated during sequence processing.

  These signals will attempt to make the outgoing weight participate in accessing the information stored in the processing unit and, at different times, protect the subsequent     unit from being perturbed by the output of the unit being fed forward.

  These conflicts are not specific to long-term lags and can equally impinge on short-term lags. Of note though is that as lag increases, stored information must be protected      from perturbation, especially in the advanced stages of learning.

- **Network Architecture:** Different types of units may convey useful information about the current state of the network. For instance, an input gate (output gate) may use        inputs from other memory cells to decide whether to store (access) certain information in its memory cell.

   Memory cells contain gates. Gates are specific to the connection they mediate. Input gates work to remedy the Input Weight Conflict while Output Gates work to eliminate the      Output Weight Conflict.

- **Gates:** Specifically, to alleviate the input and output weight conflicts and perturbations, a multiplicative input gate unit is introduced to protect the memory contents      stored from perturbation by irrelevant inputs and a multiplicative output gate unit protects other units from perturbation by currently irrelevant memory contents stored.

  Connectivity in LSTM is complicated compared to the multilayer Perceptron because of the diversity of processing elements and the inclusion of feedback connections.

- **Memory cell blocks:** Memory cells sharing the same input gate and the same output gate form a structure called a “memory cell block”.

  Memory cell blocks facilitate information storage; as with conventional neural nets, it is not so easy to code a distributed input within a single cell. A memory cell block of   size 1 is just a simple memory cell.

- **Learning:** A variant of Real Time Recurrent Learning (RTRL) that takes into account the altered, multiplicative dynamics caused by input and output gates is used to ensure    non-decaying error back propagated through internal states of memory cells errors arriving at “memory cell net inputs” do not get propagated back further in time.

- **Guessing:** This stochastic approach can outperform many term lag algorithms. It has been established that many long-time lag tasks used in previous work can be solved more    quickly by simple random weight guessing than by the proposed algorithms.


