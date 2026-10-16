# Semi-Supervised GNN Fake Account Detection (Graph Message Passing)
## Project Summary
We simulate a social network with **1000 user accounts**.
- Nodes = user accounts
- Edges = follow/interaction between users
- Node features: 6 behavioral features for every user (auto-generated randomly)
- **Only 20 users manually labeled** (Real/Fake). All other 980 users are unlabeled during training.
- Model: GCN (Graph Convolutional Network) using **graph message passing**.
Message passing: labeled nodes send information to their connected neighbors across the graph, so the model can predict labels for unlabeled users.

This semi-supervised approach mimics real-world scenario: manual labelling is costly, we only label a small sample.
This method can be transferred to NG-IoT /6G networks for malicious IoT device detection.

## Features for each user
1. avg_post_length
2. post_frequency
3. follower_count
4. following_count
5. account_age
6. number_of_comments

## How to run
1. Install dependencies
```bash
pip install -r requirements.txt