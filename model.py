import numpy as np
import matplotlib.pyplot as plt
import os
import joblib 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# ============================================
# UPDATE THIS PATH TO YOUR LOCAL EMNIST FOLDER
# ============================================
DATA_PATH = r"F:\IDE\Python\PyP\emnist_source_files"


print("Loading features from emnist_features.npz ...")
data = np.load(os.path.join(DATA_PATH, 'emnist_features.npz'))
X_train = data['X_train']
X_test  = data['X_test']
y_train = data['y_train']
y_test  = data['y_test']
print(f"  X_train shape : {X_train.shape}")
print(f"  X_test  shape : {X_test.shape}")


TRAIN_SIZE = 24960
TEST_SIZE  = 4160

print(f"\nUsing subset for speed: {TRAIN_SIZE} train, {TEST_SIZE} test samples")

np.random.seed(42)
train_idx = np.random.choice(len(X_train), TRAIN_SIZE, replace=False)
test_idx  = np.random.choice(len(X_test),  TEST_SIZE,  replace=False)

X_tr = X_train[train_idx]
y_tr = y_train[train_idx]
X_te = X_test[test_idx]
y_te = y_test[test_idx]
print(f"  Subset X_tr shape : {X_tr.shape}")
print(f"  Subset X_te shape : {X_te.shape}")


print("\n" + "="*50)
print("TUNING kNN — Testing different k values...")
print("="*50)
k_values   = [1, 3, 5, 7, 9, 11, 15]
knn_train_accuracies = []
knn_test_accuracies  = []
for k in k_values:
    print(f"  Training kNN with k={k} ...", end=' ')
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1)
    knn.fit(X_tr, y_tr)

    train_acc = accuracy_score(y_tr, knn.predict(X_tr))
    test_acc  = accuracy_score(y_te, knn.predict(X_te))

    knn_train_accuracies.append(train_acc)
    knn_test_accuracies.append(test_acc)

    print(f"Train: {train_acc:.4f} | Test: {test_acc:.4f}")

best_k_idx = np.argmax(knn_test_accuracies)
best_k     = k_values[best_k_idx]
best_knn_acc = knn_test_accuracies[best_k_idx]
print(f"\n  Best k = {best_k} with test accuracy = {best_knn_acc:.4f}")

print(f"\nTraining final kNN model with best k={best_k} ...")
best_knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean', n_jobs=-1)
best_knn.fit(X_tr, y_tr)
print("  Done.")


plt.figure(figsize=(8, 5))
plt.plot(k_values, knn_train_accuracies, 'o-', color='steelblue',
         label='Train Accuracy', linewidth=2, markersize=7)
plt.plot(k_values, knn_test_accuracies, 's--', color='coral',
         label='Test Accuracy', linewidth=2, markersize=7)
plt.axvline(x=best_k, color='green', linestyle=':', linewidth=1.5,
            label=f'Best k = {best_k}')
plt.title('kNN — Accuracy vs Number of Neighbors (k)', fontsize=13)
plt.xlabel('k (number of neighbors)')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DATA_PATH, 'knn_accuracy_vs_k.png'), dpi=150)
plt.show()
print("  Saved: knn_accuracy_vs_k.png")


print("\n" + "="*50)
print("TUNING Decision Tree — Testing different max_depth values...")
print("="*50)
depth_values = [2, 4, 6, 8, 10, 12, 14, 15, 20, None]
dt_train_accuracies = []
dt_test_accuracies  = []
for depth in depth_values:
    print(f"  Training Decision Tree with max_depth={depth} ...", end=' ')
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_tr, y_tr)

    train_acc = accuracy_score(y_tr, dt.predict(X_tr))
    test_acc  = accuracy_score(y_te, dt.predict(X_te))

    dt_train_accuracies.append(train_acc)
    dt_test_accuracies.append(test_acc)

    print(f"Train: {train_acc:.4f} | Test: {test_acc:.4f}")

best_depth_idx = np.argmax(dt_test_accuracies)
best_depth     = depth_values[best_depth_idx]
best_dt_acc    = dt_test_accuracies[best_depth_idx]
print(f"\n  Best max_depth = {best_depth} with test accuracy = {best_dt_acc:.4f}")

print(f"\nTraining final Decision Tree with best max_depth={best_depth} ...")
best_dt = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
best_dt.fit(X_tr, y_tr)
print("  Done.")


depth_labels = [str(d) if d is not None else 'None\n(unlimited)' for d in depth_values]
plt.figure(figsize=(9, 5))
plt.plot(range(len(depth_values)), dt_train_accuracies, 'o-', color='steelblue',
         label='Train Accuracy', linewidth=2, markersize=7)
plt.plot(range(len(depth_values)), dt_test_accuracies, 's--', color='coral',
         label='Test Accuracy', linewidth=2, markersize=7)
plt.axvline(x=best_depth_idx, color='green', linestyle=':', linewidth=1.5,
            label=f'Best depth = {best_depth}')
plt.title('Decision Tree — Accuracy vs Max Depth', fontsize=13)
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.xticks(range(len(depth_values)), depth_labels)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DATA_PATH, 'dt_accuracy_vs_depth.png'), dpi=150)
plt.show()
print("  Saved: dt_accuracy_vs_depth.png")


print("\nSaving trained models ...")
knn_path = os.path.join(DATA_PATH, 'best_knn_model.joblib')
dt_path  = os.path.join(DATA_PATH, 'best_dt_model.joblib')
joblib.dump(best_knn, knn_path)
joblib.dump(best_dt,  dt_path)
print("  Saved: best_knn_model.joblib")
print("  Saved: best_dt_model.joblib")

subset_path = os.path.join(DATA_PATH, 'emnist_subset.npz')
np.savez_compressed(subset_path,
                    X_tr=X_tr, y_tr=y_tr,
                    X_te=X_te, y_te=y_te)
print("  Saved: emnist_subset.npz")


print("\n" + "="*50)
print("MODEL TRAINING COMPLETE — SUMMARY")
print("="*50)
print(f"  kNN  — Best k         : {best_k}")
print(f"  kNN  — Test Accuracy  : {best_knn_acc:.4f} ({best_knn_acc*100:.2f}%)")
print(f"  DT   — Best max_depth : {best_depth}")
print(f"  DT   — Test Accuracy  : {best_dt_acc:.4f} ({best_dt_acc*100:.2f}%)")
if best_knn_acc > best_dt_acc:
    print("\n  kNN performs better on this subset.")
else:
    print("\n  Decision Tree performs better on this subset.")
print("="*50)
print("\nNext step: Run evaluation.py")
