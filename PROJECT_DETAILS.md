\# Semi-Supervised GNN Fake Account Detector - Full Project Deep Explanation

\## Project Overview

This project implements a Graph Neural Network (GCN, Graph Convolutional Network) for semi-supervised fake social media account detection. We create a synthetic dataset of 1000 user accounts. Only \*\*20 users are manually labelled\*\*, and the model predicts whether the remaining 980 accounts are fake or real.



The core idea of GNN is \*\*message passing\*\*: each user (graph node) collects feature information from its connected neighbours in the social graph. Fake accounts have different connection patterns; fake accounts tend to follow many random accounts, while real users have fewer, more stable connections. The graph structure + user features are used together for classification.



This work is relevant for next-generation IoT and 6G networks because 6G and NG-IoT will manage billions of connected devices, many of which can be malicious, spoofed, or fake IoT nodes. The same GNN message-passing framework used here to detect fake user accounts can be adapted to detect compromised or fake IoT devices in 6G massive IoT networks.



\## Dataset Details

We generate synthetic data for 1000 users.

Each user node has 6 features:

1\. avg\_post\_length: average length of posts

2\. post\_frequency: how often the user posts content

3\. follower\_count: number of followers

4\. following\_count: number of accounts this user follows

5\. account\_age: age of account

6\. number\_of\_comments: total comments made by user



We create 250 fake accounts and 750 real accounts. Real and fake users have different statistical distributions for these six features.



\### Graph Construction

We build an adjacency matrix that defines connections between users:

\- Real users: connect to 3–9 other users

\- Fake users: connect to 20–39 random users. Fake accounts create many shallow connections, which is their signature pattern.



\## GCN Mathematics \& Normalization

We use standard GCN normalization.

1\. Add self-loop: $ \\tilde{A}=A+I $

&#x20;   We add self connection for every node, so a node always sees its own features when aggregating neighbour messages.

2\. Degree matrix D: diagonal matrix where $D\_{ii}$ = total number of connections of node i

3\. Normalized adjacency matrix: $ \\hat{A}=D^{-1/2}\\tilde{A}D^{-1/2} $

Normalization prevents feature values from exploding when we aggregate messages from many neighbours.



\## GCN Layer: Message Passing

The GCN layer formula:

$$X'=\\hat{A}XW$$

Where:

\- X: node feature matrix

\- $\\hat{A}$: normalized adjacency matrix

\- W: learnable weight matrix



Step by step message passing:

1\. Each node gathers feature data from all its graph neighbours ($\\hat{A}X$).

2\. Multiply by trainable weight matrix W to transform aggregated features.

3\. Apply ReLU activation function for non-linearity.



Our model has two GCN layers:

1\. Layer 1: maps input 6 features → 32 hidden dimension embeddings.

2\. Layer 2: maps 32 hidden embeddings → 2 output classes (real / fake).

We use log softmax on final output for classification.



\## Semi-Supervised Training Setup

Only first 20 nodes have known labels (`NUM\_LABELED = 20`).

Loss function: Negative Log-Likelihood (NLL loss). We only calculate loss using these manually labelled 20 nodes. The model learns the graph structure and feature patterns from the full graph, even though we do not know labels for other nodes.

Optimizer: Adam optimizer, learning rate = 0.01, total training epochs =120.



\## Output Results Explanation

\### 1. Confusion Matrix (`confusion\_matrix.png`)

Confusion matrix shows prediction performance, rows = true labels, columns = predicted labels:

\- True Negative (TN): Real account predicted as Real

\- False Positive (FP): Real account predicted as Fake

\- False Negative (FN): Fake account predicted as Real

\- True Positive (TP): Fake account predicted as Fake



From these four values we calculate:

1\. Accuracy = (TP + TN) / (TP+TN+FP+FN)

2\. Precision = TP/(TP+FP): How many predicted fake accounts are truly fake

3\. Recall = TP/(TP+FN): How many real fake accounts we successfully detect

4\. F1-Score: balance of precision and recall



\### 2. Node Embedding Plot (`embedding.png`)

The embedding plot visualizes the 2-dimensional final representation of every user node after the GNN processes the graph.

Blue = real users, Red = fake users.

If model works well, red points and blue points will form separate clusters. The GNN learns to map similar nodes into nearby points in embedding space.



\### 3. user\_predictions.csv

This CSV stores all data: user features, ground truth label, model predicted label, and flag showing whether the node was manually labelled.



\---

\# Hyperparameter Tuning Section (How to improve model learning)

You can edit these parameters inside the code to change model behaviour:



\## 1. NUM\_LABELED (number of manually labelled nodes)

Current value: `NUM\_LABELED = 20`

\- Increase NUM\_LABELED (example: 50, 100): More manually labelled samples → model learns better, accuracy increases.

&#x20; More labelled data gives stronger loss signal during training.

\- Decrease NUM\_LABELED (like 5): Less labelled data, harder task, accuracy drops. The model has very little ground truth to learn from.



> Experiment suggestion: test NUM\_LABELED = 20, 50, 100 and compare final accuracy and confusion matrix.



\## 2. Learning Rate `lr=0.01`

Current: `opt = torch.optim.Adam(model.parameters(), lr=0.01)`

\- Higher lr (0.05): Faster initial learning. Risk: model overshoot optimal values, unstable loss, may not converge.

\- Lower lr (0.001): Training is slower, updates are tiny. More stable, needs more epochs to finish learning.



\## 3. Epochs `epochs=120`

Epochs = number of full passes over graph data during training.

\- Too small (20): model underfits, not enough time to learn patterns. Low accuracy.

\- Too large (300+): risk of overfitting. Model memorizes the small labelled nodes and performs worse on unlabeled nodes.



\## 4. Hidden dimension size `hidden\_dim=32`

In `GCN(NUM\_FEATURES, 32, 2)`

\- Larger hidden dimension (64, 128): model can learn more complex patterns. More parameters, higher computation cost, risk of overfit.

\- Smaller hidden dimension (8, 16): simpler model, less capacity, may underfit.



\## 5. Number of GCN layers

Currently two layers: l1 and l2.

\- 1 layer: simpler, less expressive

\- 3 or more layers: \*\*over-smoothing problem\*\*. All node embeddings become identical, model loses classification ability. For semi-supervised node classification, 2 layers is usually best choice.



\## 6. Graph connection rules

You can modify how neighbours are created. Change the random neighbour count for fake and real users to make the problem harder or easier.



\---

\# Linkage to 6G / Next Generation NG-IoT Networks

In 6G massive IoT, millions or billions of IoT devices connect in a large-scale network. Attackers can create fake/spoofed IoT nodes to launch network attacks.

Each IoT device can be treated as a graph node, wireless communication links between devices are graph edges.

The GNN message-passing method in this project can be adapted for 6G IoT security tasks:

1\. Detect fake / compromised IoT devices by analysing device features and wireless connection graph.

2\. Semi-supervised learning is very useful for 6G IoT, because manually labelling huge numbers of IoT devices is expensive. We only label a small subset of devices, and the GNN detects malicious devices across the whole network.

3\. 6G networks have high requirements for low latency. The GCN graph aggregation can be optimized to run efficiently on edge computing hardware deployed inside the 6G base station.

4\. This same spectral graph theory used for GCN can also connect to LDPC and belief propagation algorithms you are researching for your survey paper. Both use message passing over graphs for distributed inference.



\## Limitations of this demo project

1\. Dataset is synthetic. Real-world IoT / social network data has noise and missing features.

2\. Static graph: the graph connections never change during training. Real 6G IoT networks are dynamic, devices join and leave continuously.

3\. Simple 2-layer GCN. Advanced GNN variants (GAT, GraphSAGE) can handle dynamic networks better.



\## Future Work

1\. Adapt the static GCN to dynamic graphs for time-varying 6G IoT connections.

2\. Test with real IoT network traces instead of synthetic data.

3\. Integrate with LDPC and belief propagation for joint detection and error correction in 6G.

4\. Add attention mechanism (GAT) to weight important neighbour messages higher.





\### GNN vs GCN Clarification

Graph Neural Network (GNN) is the broad family of neural networks designed for graph data.

Graph Convolutional Network (GCN) is one specific member inside the GNN family.

This project implements a 2-layer GCN. It follows the core GNN principle: \*\*message passing\*\*.

Each node aggregates feature information from its neighbouring nodes on the graph.



GCN uses normalized graph adjacency matrix to perform the aggregation step, which is its unique characteristic.

Other GNN architectures (such as GAT, GraphSAGE) use different aggregation rules, but they all belong to the GNN family.



When we describe the project in high-level writing (research abstract, introduction), we can refer to it as a GNN-based detector.

When discussing model implementation and mathematics, we specify it is a GCN model.



