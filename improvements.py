import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score


# ============================================
# UPDATE THIS PATH TO YOUR LOCAL EMNIST FOLDER
# ============================================
DATA_PATH = r"F:\IDE\Python\PyP\emnist_source_files"

alphabet   = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')


print("Loading models and data...")
best_knn = joblib.load(os.path.join(DATA_PATH, 'best_knn_model.joblib'))
best_dt  = joblib.load(os.path.join(DATA_PATH, 'best_dt_model.joblib'))
subset   = np.load(os.path.join(DATA_PATH, 'emnist_subset.npz'))
X_tr     = subset['X_tr']
y_tr     = subset['y_tr']
X_te     = subset['X_te']
y_te     = subset['y_te']

clean    = np.load(os.path.join(DATA_PATH, 'emnist_letters_clean.npz'))

knn_pred = best_knn.predict(X_te)
dt_pred  = best_dt.predict(X_te)
print("  Everything loaded.")


print("\nAnalyzing Decision Tree feature importance...")
importances    = best_dt.feature_importances_  
importance_map = importances.reshape(28, 28)

plt.figure(figsize=(7, 6))
plt.imshow(importance_map, cmap='hot')
plt.colorbar(label='Feature Importance (higher = more used by Decision Tree)')
plt.title('Decision Tree — Feature Importance Map\nWhich pixels the tree uses most for splitting')
plt.xlabel('Pixel column')
plt.ylabel('Pixel row')
plt.tight_layout()
plt.savefig(os.path.join(DATA_PATH, 'dt_feature_importance.png'), dpi=150)
plt.show()
print("  Saved: dt_feature_importance.png")

top_pixels = np.argsort(importances)[::-1][:10]
print("\n  Top 10 most important pixels (by position):")
for rank, pixel_idx in enumerate(top_pixels):
    row = pixel_idx // 28
    col = pixel_idx % 28
    print(f"    Rank {rank+1}: pixel_{pixel_idx} "
          f"(row={row}, col={col}) "
          f"importance={importances[pixel_idx]:.6f}")

 
print("\nGenerating average image per class...")
fig, axes = plt.subplots(4, 7, figsize=(14, 8))
fig.suptitle('Average Image per Class — What the Model Sees as Typical Letter',
             fontsize=13)

axes_flat = axes.flatten()
for class_idx in range(26):
    indices    = np.where(y_tr == class_idx)[0]
    avg_image  = X_tr[indices].mean(axis=0).reshape(28, 28)
    axes_flat[class_idx].imshow(avg_image, cmap='gray')
    axes_flat[class_idx].set_title(alphabet[class_idx], fontsize=11)
    axes_flat[class_idx].axis('off')

for idx in range(26, 28):
    axes_flat[idx].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(DATA_PATH, 'average_image_per_class.png'), dpi=150)
plt.show()
print("  Saved: average_image_per_class.png")


print("\nAnalyzing top confused letter pairs...")
cm_knn     = confusion_matrix(y_te, knn_pred)
cm_no_diag = cm_knn.copy()
np.fill_diagonal(cm_no_diag, 0)

confused_pairs = []
for i in range(26):
    for j in range(26):
        if i != j and cm_no_diag[i, j] > 0:
            confused_pairs.append((cm_no_diag[i, j], i, j))
confused_pairs.sort(reverse=True)
top_confused = confused_pairs[:6]

fig, axes = plt.subplots(6, 3, figsize=(8, 16))
fig.suptitle('Top 6 Most Confused Letter Pairs — Actual Misclassified Images',
             fontsize=13)

for row, (count, true_idx, pred_idx) in enumerate(top_confused):
    error_mask = (y_te == true_idx) & (knn_pred == pred_idx)
    error_indices = np.where(error_mask)[0]

    for col in range(3):
        ax = axes[row, col]
        if col < len(error_indices):
            img = X_te[error_indices[col]].reshape(28, 28)
            ax.imshow(img, cmap='gray')
            ax.set_title(
                f'True: {alphabet[true_idx]}\n'
                f'Got: {alphabet[pred_idx]}\n'
                f'({count} errors)',
                fontsize=8, color='red'
            )
        ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(DATA_PATH, 'error_analysis.png'), dpi=150)
plt.show()
print("  Saved: error_analysis.png")


print("\nPlotting per-class error rates...")
knn_per_class = np.diag(cm_knn) / np.sum(cm_knn, axis=1)
knn_error_rate = 1 - knn_per_class

cm_dt        = confusion_matrix(y_te, dt_pred)
dt_per_class = np.diag(cm_dt) / np.sum(cm_dt, axis=1)
dt_error_rate = 1 - dt_per_class

sorted_idx = np.argsort(knn_error_rate)[::-1]

fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(26)
width = 0.35

ax.bar(x - width/2,
       [knn_error_rate[i] for i in sorted_idx],
       width, label='kNN Error Rate', color='steelblue', alpha=0.8)
ax.bar(x + width/2,
       [dt_error_rate[i] for i in sorted_idx],
       width, label='DT Error Rate', color='coral', alpha=0.8)

ax.set_xlabel('Letter Class (sorted by kNN error rate)')
ax.set_ylabel('Error Rate')
ax.set_title('Per-Class Error Rate — Letters Needing Most Improvement')
ax.set_xticks(x)
ax.set_xticklabels([alphabet[i] for i in sorted_idx])
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(DATA_PATH, 'error_rate_per_class.png'), dpi=150)
plt.show()
print("  Saved: error_rate_per_class.png")


print("\nPlotting model comparison chart...")
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_gmean(y_true, y_pred):
    """Calculate Geometric Mean of per-class recall scores."""
    cm = confusion_matrix(y_true, y_pred)
    per_class_recall = np.diag(cm) / np.sum(cm, axis=1)
    per_class_recall = np.where(per_class_recall == 0, 1e-10, per_class_recall)
    return np.exp(np.mean(np.log(per_class_recall)))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'G-Mean']
knn_scores = [
    accuracy_score(y_te, knn_pred),
    precision_score(y_te, knn_pred, average='macro', zero_division=0),
    recall_score(y_te, knn_pred, average='macro', zero_division=0),
    f1_score(y_te, knn_pred, average='macro', zero_division=0),
    calculate_gmean(y_te, knn_pred)
]

dt_scores = [
    accuracy_score(y_te, dt_pred),
    precision_score(y_te, dt_pred, average='macro', zero_division=0),
    recall_score(y_te, dt_pred, average='macro', zero_division=0),
    f1_score(y_te, dt_pred, average='macro', zero_division=0),
    calculate_gmean(y_te, dt_pred)
]

x = np.arange(len(metrics))
width = 0.3

fig, ax = plt.subplots(figsize=(11, 6))
bars1 = ax.bar(x - width/2, knn_scores, width,
               label='kNN', color='steelblue', alpha=0.85)
bars2 = ax.bar(x + width/2, dt_scores, width,
               label='Decision Tree', color='coral', alpha=0.85)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height()*100:.1f}%', ha='center', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height()*100:.1f}%', ha='center', fontsize=9)

ax.set_ylabel('Score')
ax.set_title('Model Comparison — kNN vs Decision Tree (All Metrics)')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(DATA_PATH, 'model_comparison.png'), dpi=150)
plt.show()
print("  Saved: model_comparison.png")


print("\n" + "="*55)
print("KEY INSIGHTS AND FINDINGS")
print("="*55)
print("\n  1. MODEL PERFORMANCE:")
print(f"     kNN accuracy  : {accuracy_score(y_te, knn_pred)*100:.2f}%")
print(f"     DT  accuracy  : {accuracy_score(y_te, dt_pred)*100:.2f}%")
print(f"     kNN is better by : "
      f"{(accuracy_score(y_te,knn_pred)-accuracy_score(y_te,dt_pred))*100:.2f}%")

print("\n  2. BEST PERFORMING LETTERS (kNN):")
best_letters = np.argsort(knn_per_class)[::-1][:5]
for i in best_letters:
    print(f"     {alphabet[i]}: {knn_per_class[i]*100:.1f}%")

print("\n  3. WORST PERFORMING LETTERS (kNN):")
worst_letters = np.argsort(knn_per_class)[:5]
for i in worst_letters:
    print(f"     {alphabet[i]}: {knn_per_class[i]*100:.1f}%")

print("\n  4. TOP CONFUSED PAIRS (kNN):")
for count, true_idx, pred_idx in top_confused:
    print(f"     {alphabet[true_idx]} mistaken as "
          f"{alphabet[pred_idx]} — {count} times")

print("\n  5. FEATURE IMPORTANCE:")
print(f"     Most important pixel: pixel_{top_pixels[0]} "
      f"(row={top_pixels[0]//28}, col={top_pixels[0]%28})")
print("     Center pixels (rows 8-20, cols 8-20) carry most information")
print("     Edge pixels (rows 0-3, cols 0-3) carry almost no information")

print("\n  6. WHY kNN BEATS DECISION TREE:")
print("     kNN compares entire 784-pixel vectors at once")
print("     Decision Tree makes sequential single-pixel decisions")
print("     Spatial pixel relationships favor kNN for image data")

print("="*55)
print("\nNext step: Run drawing_gui.py")
