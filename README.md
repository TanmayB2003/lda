# Latent Dirichlet Allocation
## Introduction
Latent Dirichlet Allocation (LDA) is a statistical model used for topic modeling, where each document is assumed to be a mixture of topics and each word is attributed to one of these topics. In LDA, Gibbs Sampling is a technique used for probabilistic inference to estimate the hidden structure of topics within the documents.

Gibbs Sampling in LDA works by iteratively updating the assignment of words to topics. At each iteration, a word's topic assignment is re-sampled based on the current state of the model. This process is repeated many times until convergence.

Here's how Gibbs Sampling works in the context of LDA:

1. **Initialization**: Start by randomly assigning each word in each document to one of the topics.

2. **Iterative Process**:
   - For each word in each document:
     - Calculate the probability of the word belonging to each topic, considering the current state of the model.
     - Sample a new topic assignment for the word based on these probabilities.

3. **Repeat**: Repeat the above step for many iterations until the model converges to a stable state.

4. **Estimation**: After convergence, the distribution of topics in each document and the distribution of words in each topic are estimated based on the sampled assignments.

Gibbs Sampling allows LDA to iteratively refine its estimates of topics and their distributions in the documents. It works well even when the exact mathematical solution is complex or intractable.

In summary, Gibbs Sampling in LDA is a powerful technique for estimating the latent structure of topics within documents by iteratively updating the assignments of words to topics based on the observed data.

## Steps to run

In this implementation of LDA leveraging Gibbs Sampling, we'll be sampling words of 50 Wikipedia pages under the Category-'History', containing 14351 unique words and distributing the words into 10 topics. We'll be running 5 and 100 iterations of the sampling to distribute the words into topics.

To run the code:
1. Clone the repository
   ```bash
    git clone https://github.com/sima-sta14/lda.git
   ```
2. Install the dependencies
   ```bash
    pip install numpy matplotlib seaborn wikipediaapi nltk
   ```  
3. Run the script
   ```bash
    python3 gibbs.py
   ```
## Results
For number of iterations = 5
![graph1](https://github.com/sima-sta14/lda/assets/106529298/09bd90c3-a49b-470d-91d8-2ae2f81d1c0f)

For number of iterations = 100
![image](https://github.com/sima-sta14/lda/assets/106529298/d28c7f52-e384-45a0-9aea-8e8257d2024b)

