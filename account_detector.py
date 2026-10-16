"""
Semi-Supervised GNN Fake Account Detector
Pure torch only, NO torch_geometric required
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import pandas as pd

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fix random seed
np.random.seed(42)
torch.manual_seed(42)

NUM_USERS = 1000
NUM_FEATURES = 6
NUM_LABELED = 20

# Generate synthetic dataset
true_label = np.zeros(NUM_USERS, dtype=np.int64)
fake_users = np.random.choice(NUM_USERS, size=250, replace=False)
true_label[fake_users] = 1

features = np.zeros((NUM_USERS, NUM_FEATURES))
for i in range(NUM_USERS):
    if true_label[i] == 0:
        feat = np.random.normal(loc=[50, 3, 220, 180, 700, 12], scale=[15, 1, 90, 70, 200, 4])
    else:
        feat = np.random.normal(loc=[12, 12, 15, 900, 45, 1], scale=[4, 3, 8, 200, 15, 0.5])
    features[i] = feat

# Build graph adjacency matrix
adj = np.zeros((NUM_USERS, NUM_USERS))
for i in range(NUM_USERS):
    if true_label[i]==0:
        n = np.random.randint(3,10)
    else:
        n = np.random.randint(20,40)
    neighbours = np.random.choice(NUM_USERS, size=n, replace=False)
    adj[i, neighbours] = 1

# GCN normalization
A_tilde = adj + np.eye(NUM_USERS)
D = np.diag(np.sum(A_tilde, axis=1))
D_sqrt_inv = np.diag(1.0 / (np.sqrt(np.diag(D)) + 1e-8))
A_norm = D_sqrt_inv @ A_tilde @ D_sqrt_inv

# Convert to torch tensors
x = torch.tensor(features, dtype=torch.float32)
A_norm_t = torch.tensor(A_norm, dtype=torch.float32)
y = torch.tensor(true_label, dtype=torch.long)

# Train mask: only first 20 nodes labeled
train_mask = torch.zeros(NUM_USERS, dtype=torch.bool)
train_mask[:NUM_LABELED] = True

print("=== Fake Account Semi-Supervised GNN ===")
print(f"Total users: {NUM_USERS}")
print(f"Manually labeled users: {NUM_LABELED}")
print(f"Unlabeled users to predict: {NUM_USERS-NUM_LABELED}")

# Custom GCN Layer
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w = nn.Linear(in_dim, out_dim)
    def forward(self, x, adj):
        x = adj @ x
        x = self.w(x)
        return x

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.l1 = GCNLayer(in_dim, hidden_dim)
        self.l2 = GCNLayer(hidden_dim, out_dim)
    def forward(self, x, adj):
        x = F.relu(self.l1(x, adj))
        x = self.l2(x, adj)
        return F.log_softmax(x, dim=1)

model = GCN(NUM_FEATURES, 32, 2)
opt = torch.optim.Adam(model.parameters(), lr=0.01)

# Train
epochs = 120
for e in range(1, epochs+1):
    model.train()
    opt.zero_grad()
    pred = model(x, A_norm_t)
    loss = F.nll_loss(pred[train_mask], y[train_mask])
    loss.backward()
    opt.step()
    if e % 10 == 0:
        y_pred = pred.argmax(dim=1)
        acc = accuracy_score(y.numpy(), y_pred.detach().numpy())
        print(f"Epoch {e:3d} | Loss: {loss.item():.4f} | Accuracy: {acc:.4f}")

# Final prediction
model.eval()
with torch.no_grad():
    out = model(x, A_norm_t)
y_pred = out.argmax(dim=1).numpy()
final_acc = accuracy_score(true_label, y_pred)
print(f"\nFinal Accuracy: {final_acc:.4f}")

# Save CSV
df = pd.DataFrame(features, columns=[
    "avg_post_length",
    "post_frequency",
    "follower_count",
    "following_count",
    "account_age",
    "number_of_comments"
])
df["user_id"] = np.arange(NUM_USERS)
df["true_is_fake"] = true_label
df["predicted_is_fake"] = y_pred
df["manually_labeled"] = False
df.loc[:NUM_LABELED-1, "manually_labeled"] = True
df.to_csv(os.path.join(OUTPUT_DIR,"user_predictions.csv"), index=False)
print("✅ CSV saved to outputs/user_predictions.csv")

# Confusion matrix
cm = confusion_matrix(true_label, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Real","Fake"])
disp.plot()
plt.title("Confusion Matrix Semi-Supervised GNN")
plt.savefig(os.path.join(OUTPUT_DIR,"confusion_matrix.png"), dpi=300)
plt.show()

# Embedding plot
emb = out.detach().numpy()
plt.figure()
plt.scatter(emb[true_label==0,0], emb[true_label==0,1], c="blue", alpha=0.6, label="Real")
plt.scatter(emb[true_label==1,0], emb[true_label==1,1], c="red", alpha=0.6, label="Fake")
plt.legend()
plt.title("GNN Node Embeddings")
plt.savefig(os.path.join(OUTPUT_DIR,"embedding.png"), dpi=300)
plt.show()
print("\n✅ Program complete, all files saved to outputs folder.")