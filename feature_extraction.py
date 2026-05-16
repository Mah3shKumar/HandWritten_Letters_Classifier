import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================
# UPDATE THIS PATH TO YOUR LOCAL EMNIST FOLDER
# ============================================
DATA_PATH = r"F:\IDE\Python\PyP\emnist_source_files"


print("Loading cleaned data from emnist_letters_clean.npz ...")
data = np.load(os.path.join(DATA_PATH, 'emnist_letters_clean.npz'))
train_images = data['train_images']  
train_labels = data['train_labels']  
test_images  = data['test_images']   
test_labels  = data['test_labels']   
print(f"  Train images shape : {train_images.shape}")
print(f"  Test  images shape : {test_images.shape}")


print("\nFlattening images into feature vectors...")
X_train = train_images.reshape(-1, 784)
X_test  = test_images.reshape(-1, 784)
y_train = train_labels
y_test  = test_labels
print(f"  X_train shape : {X_train.shape}  (samples x features)")
print(f"  X_test  shape : {X_test.shape}")
print(f"  y_train shape : {y_train.shape}")
print(f"  y_test  shape : {y_test.shape}")


print("\nVisualizing flattening process...")
alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle('Feature Extraction — From Image to Feature Vector', fontsize=13)

sample_idx = 0
axes[0].imshow(train_images[sample_idx], cmap='gray')
axes[0].set_title(f'Original Image\n28x28 pixels\nLabel: {alphabet[y_train[sample_idx]]}')
axes[0].axis('off')

axes[1].imshow(train_images[sample_idx], cmap='Blues')
axes[1].set_title('Pixel Values\n(each cell = one feature)\ndarker = higher value')
for i in range(0, 28, 4):
    for j in range(0, 28, 4):
        val = train_images[sample_idx][i, j]
        axes[1].text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=5, color='white' if val > 0.5 else 'black')
axes[1].axis('off')

flat = X_train[sample_idx]
axes[2].plot(flat, color='steelblue', linewidth=0.5)
axes[2].fill_between(range(784), flat, alpha=0.3, color='steelblue')
axes[2].set_title('Flattened Feature Vector\n784 pixel values in a single row')
axes[2].set_xlabel('Feature index (pixel position 0–783)')
axes[2].set_ylabel('Pixel value [0, 1]')
axes[2].set_xlim(0, 784)
axes[2].set_ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(DATA_PATH, 'feature_extraction_visual.png'), dpi=150)
plt.show()
print("  Saved: feature_extraction_visual.png")


print("\nCreating feature table (first 5 rows, first 10 features)...")
feature_cols = [f'pixel_{i}' for i in range(784)]
df_sample = pd.DataFrame(X_train[:1000], columns=feature_cols)
df_sample['label'] = [alphabet[l] for l in y_train[:1000]]
display_cols = feature_cols[:10] + ['label']
print(df_sample[display_cols].head())
print(f"\n  Full feature table shape: {df_sample.shape}")
print("  Each row = 1 image, Each column = 1 pixel feature")

csv_path = os.path.join(DATA_PATH, 'feature_table_sample.csv')
df_sample.to_csv(csv_path, index=False)
print("  Saved: feature_table_sample.csv (first 1000 samples)")


print("\nCalculating pixel variance across all training samples...")
pixel_variance = np.var(X_train, axis=0)

variance_map = pixel_variance.reshape(28, 28)

plt.figure(figsize=(7, 6))
plt.imshow(variance_map, cmap='hot')
plt.colorbar(label='Variance (higher = more informative pixel)')
plt.title('Pixel Variance Map\nBrighter pixels carry more information for classification')
plt.xlabel('Pixel column')
plt.ylabel('Pixel row')
plt.tight_layout()
plt.savefig(os.path.join(DATA_PATH, 'pixel_variance_map.png'), dpi=150)
plt.show()
print("  Saved: pixel_variance_map.png")


print("\nSaving feature data to emnist_features.npz ...")
features_path = os.path.join(DATA_PATH, 'emnist_features.npz')
np.savez_compressed(features_path,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test)
print(f"  Saved to: {features_path}")


print("\n" + "="*50)
print("FEATURE EXTRACTION COMPLETE — SUMMARY")
print("="*50)
print(f"  Training samples  : {X_train.shape[0]}")
print(f"  Test samples      : {X_test.shape[0]}")
print(f"  Features per image: {X_train.shape[1]} (28x28 pixels flattened)")
print(f"  Classes           : {len(np.unique(y_train))} letters (A-Z)")
print(f"  Feature range     : [{X_train.min():.1f}, {X_train.max():.1f}]")
print("="*50)
print("\nNext step: Run models.py")
