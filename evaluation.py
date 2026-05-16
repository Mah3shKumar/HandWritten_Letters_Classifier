import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score,
                             classification_report)


# ============================================
# UPDATE THIS PATH TO YOUR LOCAL EMNIST FOLDER
# ============================================
DATA_PATH = r"F:\IDE\Python\PyP\emnist_source_files"

alphabet  = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')


print("Loading models and data...")
best_knn = joblib.load(os.path.join(DATA_PATH, 'best_knn_model.joblib'))
best_dt  = joblib.load(os.path.join(DATA_PATH, 'best_dt_model.joblib'))
subset   = np.load(os.path.join(DATA_PATH, 'emnist_subset.npz'))
X_tr     = subset['X_tr']
y_tr     = subset['y_tr']
X_te     = subset['X_te']
y_te     = subset['y_te']
print(f"  Test samples : {len(X_te)}")
print("  Models loaded: kNN and Decision Tree")


print("\nGenerating predictions...")
knn_pred = best_knn.predict(X_te)
dt_pred  = best_dt.predict(X_te)
print("  Predictions done.")


print("\n" + "="*50)
print("EVALUATION METRICS")
print("="*50)
def calculate_gmean(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    per_class_recall = np.diag(cm) / np.sum(cm, axis=1)
    per_class_recall = np.where(per_class_recall == 0, 1e-10, per_class_recall)
    gmean = np.exp(np.mean(np.log(per_class_recall)))
    return gmean

for model_name, y_pred in [('kNN', knn_pred), ('Decision Tree', dt_pred)]:
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_te, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_te, y_pred, average='macro', zero_division=0)
    gm   = calculate_gmean(y_te, y_pred)

    print(f"\n  {model_name}:")
    print(f"    Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"    Precision : {prec:.4f} ({prec*100:.2f}%)")
    print(f"    Recall    : {rec:.4f}  ({rec*100:.2f}%)")
    print(f"    F1-Score  : {f1:.4f}  ({f1*100:.2f}%)")
    print(f"    G-Mean    : {gm:.4f}  ({gm*100:.2f}%)")


print("\nPlotting confusion matrix for kNN...")
fig, ax = plt.subplots(figsize=(16, 14))
cm_knn = confusion_matrix(y_te, knn_pred)

im = ax.imshow(cm_knn, cmap='Blues')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax.set_xticks(range(26))
ax.set_yticks(range(26))
ax.set_xticklabels(alphabet, fontsize=10)
ax.set_yticklabels(alphabet, fontsize=10)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('kNN — Confusion Matrix (Aa–Zz Classification)', fontsize=14)

for i in range(26):
    for j in range(26):
        val = cm_knn[i, j]
        if val > 0:
            color = 'white' if val > cm_knn.max() * 0.5 else 'black'
            ax.text(j, i, str(val), ha='center', va='center',
                   fontsize=6, color=color)

plt.tight_layout()
plt.savefig(os.path.join(DATA_PATH, 'confusion_matrix_knn.png'), dpi=150)
plt.show()
print("  Saved: confusion_matrix_knn.png")


print("\nPlotting confusion matrix for Decision Tree...")

fig, ax = plt.subplots(figsize=(16, 14))
cm_dt = confusion_matrix(y_te, dt_pred)

im = ax.imshow(cm_dt, cmap='Oranges')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax.set_xticks(range(26))
ax.set_yticks(range(26))
ax.set_xticklabels(alphabet, fontsize=10)
ax.set_yticklabels(alphabet, fontsize=10)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('Decision Tree — Confusion Matrix (Aa–Zz Classification)', fontsize=14)

for i in range(26):
    for j in range(26):
        val = cm_dt[i, j]
        if val > 0:
            color = 'white' if val > cm_dt.max() * 0.5 else 'black'
            ax.text(j, i, str(val), ha='center', va='center',
                   fontsize=6, color=color)

plt.tight_layout()
plt.savefig(os.path.join(DATA_PATH, 'confusion_matrix_dt.png'), dpi=150)
plt.show()
print("  Saved: confusion_matrix_dt.png")


print("\n" + "="*50)
print("TOP 10 MOST CONFUSED LETTER PAIRS")
print("="*50)
for model_name, cm in [('kNN', cm_knn), ('Decision Tree', cm_dt)]:
    print(f"\n  {model_name}:")
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)

    confused_pairs = []
    for i in range(26):
        for j in range(26):
            if i != j and cm_no_diag[i, j] > 0:
                confused_pairs.append((cm_no_diag[i, j], alphabet[i], alphabet[j]))

    confused_pairs.sort(reverse=True)

    for count, true_lbl, pred_lbl in confused_pairs[:10]:
        print(f"    True: {true_lbl} → Predicted: {pred_lbl}  ({count} times)")


print("\nPlotting per-class accuracy for kNN...")

knn_per_class = np.diag(cm_knn) / np.sum(cm_knn, axis=1)
dt_per_class  = np.diag(cm_dt)  / np.sum(cm_dt,  axis=1)

x = np.arange(26)
width = 0.35

fig, ax = plt.subplots(figsize=(16, 6))
bars1 = ax.bar(x - width/2, knn_per_class, width,
               label='kNN', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, dt_per_class, width,
               label='Decision Tree', color='coral', alpha=0.8)

ax.set_xlabel('Letter Class (A–Z)')
ax.set_ylabel('Per-Class Accuracy')
ax.set_title('Per-Class Accuracy — kNN vs Decision Tree')
ax.set_xticks(x)
ax.set_xticklabels(alphabet)
ax.set_ylim(0, 1.1)
ax.axhline(y=knn_per_class.mean(), color='steelblue', linestyle='--',
           linewidth=1, alpha=0.6)
ax.axhline(y=dt_per_class.mean(), color='coral', linestyle='--',
           linewidth=1, alpha=0.6)
ax.legend([bars1, bars2,
           plt.Line2D([0], [0], color='steelblue', linestyle='--', linewidth=1),
           plt.Line2D([0], [0], color='coral', linestyle='--', linewidth=1)],
          ['kNN', 'Decision Tree', 'kNN Mean', 'DT Mean'])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(DATA_PATH, 'per_class_accuracy.png'), dpi=150)
plt.show()
print("  Saved: per_class_accuracy.png")


print("\nVisualizing correct and incorrect classifications...")

clean_data  = np.load(os.path.join(DATA_PATH, 'emnist_letters_clean.npz'))
all_test_images = clean_data['test_images']
all_test_labels = clean_data['test_labels']

subset_data = np.load(os.path.join(DATA_PATH, 'emnist_subset.npz'))

correct_idx   = np.where(knn_pred == y_te)[0]
incorrect_idx = np.where(knn_pred != y_te)[0]

fig, axes = plt.subplots(2, 10, figsize=(18, 5))
fig.suptitle('kNN — Correct (top) and Incorrect (bottom) Classifications',
             fontsize=13)

for col, idx in enumerate(correct_idx[:10]):
    img = X_te[idx].reshape(28, 28)
    axes[0, col].imshow(img, cmap='gray')
    axes[0, col].set_title(f'T:{alphabet[y_te[idx]]}\nP:{alphabet[knn_pred[idx]]}',
                           fontsize=8, color='green')
    axes[0, col].axis('off')

for col, idx in enumerate(incorrect_idx[:10]):
    img = X_te[idx].reshape(28, 28)
    axes[1, col].imshow(img, cmap='gray')
    axes[1, col].set_title(f'T:{alphabet[y_te[idx]]}\nP:{alphabet[knn_pred[idx]]}',
                           fontsize=8, color='red')
    axes[1, col].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(DATA_PATH, 'correct_incorrect_knn.png'), dpi=150)
plt.show()
print("  Saved: correct_incorrect_knn.png")


print("\nSaving full classification reports...")
report_path = os.path.join(DATA_PATH, 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write("="*60 + "\n")
    f.write("kNN CLASSIFICATION REPORT\n")
    f.write("="*60 + "\n")
    f.write(classification_report(y_te, knn_pred,
            target_names=alphabet, zero_division=0))
    f.write("\n\n")
    f.write("="*60 + "\n")
    f.write("DECISION TREE CLASSIFICATION REPORT\n")
    f.write("="*60 + "\n")
    f.write(classification_report(y_te, dt_pred,
            target_names=alphabet, zero_division=0))

print("  Saved: classification_report.txt")


print("\n" + "="*50)
print("EVALUATION COMPLETE — SUMMARY")
print("="*50)

knn_acc  = accuracy_score(y_te, knn_pred)
dt_acc   = accuracy_score(y_te, dt_pred)
knn_f1   = f1_score(y_te, knn_pred, average='macro', zero_division=0)
dt_f1    = f1_score(y_te, dt_pred,  average='macro', zero_division=0)
knn_gm   = calculate_gmean(y_te, knn_pred)
dt_gm    = calculate_gmean(y_te, dt_pred)

print(f"  {'Metric':<15} {'kNN':>10} {'Decision Tree':>15}")
print(f"  {'-'*42}")
print(f"  {'Accuracy':<15} {knn_acc*100:>9.2f}% {dt_acc*100:>14.2f}%")
print(f"  {'F1-Score':<15} {knn_f1*100:>9.2f}% {dt_f1*100:>14.2f}%")
print(f"  {'G-Mean':<15} {knn_gm*100:>9.2f}% {dt_gm*100:>14.2f}%")
print(f"  {'-'*42}")
print(f"\n  Winner: {'kNN' if knn_acc > dt_acc else 'Decision Tree'}")
print("="*50)
print("\nNext step: Run improvements.py")
