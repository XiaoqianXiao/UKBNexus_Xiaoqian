#%%
results_dir = '/Users/xiaoqianxiao/UKB/presentation'
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
import os

# Apply Seaborn poster style
sns.set_context("poster")  # Poster style from Seaborn (larger fonts and markers)

# Generate synthetic data with better class separability to achieve ~90% accuracy
X, y = make_classification(
    n_samples=200, n_features=10, n_informative=6, n_redundant=0,
    n_clusters_per_class=1, class_sep=2.0, flip_y=0.01, random_state=42
)

# Select the 2 most effective features using ANOVA F-statistic
selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X, y)
most_effective_features = selector.get_support(indices=True)

# Train an SVM classifier on the selected features
svm = SVC(kernel='linear', C=1)
svm.fit(X_selected, y)

# Predict on the dataset
y_pred = svm.predict(X_selected)

# Calculate accuracy
accuracy = accuracy_score(y, y_pred)

# Create the Seaborn-style plot
plt.figure(figsize=(3, 3))  # Larger figure for poster style

# Scatter plot of the data points with Seaborn styling
#sns.scatterplot(x=X_selected[:, 0], y=X_selected[:, 1], hue=y, palette='coolwarm', s=20, legend=None)
sns.scatterplot(x=X_selected[y == 0, 0], y=X_selected[y == 0, 1], color='tab:blue', marker='o', s=30)
sns.scatterplot(x=X_selected[y == 1, 0], y=X_selected[y == 1, 1], color='tab:orange', marker='x', s=30)
# Plot the decision boundary
xx = np.linspace(X_selected[:, 0].min() - 1, X_selected[:, 0].max() + 1, 100)
yy = np.linspace(X_selected[:, 1].min() - 1, X_selected[:, 1].max() + 1, 100)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm.decision_function(xy).reshape(XX.shape)

# Contour plot for the decision boundary and margins
plt.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=1.0, linestyles=['--', '-', '--'], linewidths=3)

# Add labels and legend without bold styling
#plt.title(f"Accuracy ({accuracy * 100:.2f}%)", fontsize=32)
#plt.title("Accuracy", fontsize=32)
plt.title("Accuracy")
#plt.xlabel(f"Most Effective Feature {most_effective_features[0] + 1}", fontsize=28)
#plt.ylabel(f"Most Effective Feature {most_effective_features[1] + 1}", fontsize=28)
#plt.legend(fontsize=24, loc='upper left', frameon=True)
# Remove x and y ticks
plt.xticks([])
plt.yticks([])
# Show the plot
results_path = os.path.join(results_dir, 'svm_results.png')
plt.savefig(results_path, bbox_inches='tight', dpi=300)
plt.tight_layout()
plt.show()

#%%
results_dir = '/Users/xiaoqianxiao/UKB/presentation'
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
import os

# Apply Seaborn poster style
sns.set_context("poster")  # Poster style from Seaborn (larger fonts and markers)

# Generate synthetic data with better class separability to achieve ~90% accuracy
X, y = make_classification(
    n_samples=200, n_features=10, n_informative=6, n_redundant=0,
    n_clusters_per_class=1, class_sep=2.0, flip_y=0.01, random_state=42
)

# Select the 2 most effective features using ANOVA F-statistic
selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X, y)
most_effective_features = selector.get_support(indices=True)

# Train an SVM classifier on the selected features
svm = SVC(kernel='linear', C=1)
svm.fit(X_selected, y)

# Predict on the dataset
y_pred = svm.predict(X_selected)

# Calculate accuracy
accuracy = accuracy_score(y, y_pred)

# Create the Seaborn-style plot
plt.figure(figsize=(3, 3))  # Larger figure for poster style

# Scatter plot of the data points with Seaborn styling
#sns.scatterplot(x=X_selected[:, 0], y=X_selected[:, 1], hue=y, palette='coolwarm', s=20, legend=None)
sns.scatterplot(x=X_selected[y == 0, 0], y=X_selected[y == 0, 1], color='tab:blue', marker='o', s=30)
sns.scatterplot(x=X_selected[y == 1, 0], y=X_selected[y == 1, 1], color='tab:orange', marker='x', s=30)
# Plot the decision boundary
xx = np.linspace(X_selected[:, 0].min() - 1, X_selected[:, 0].max() + 1, 100)
yy = np.linspace(X_selected[:, 1].min() - 1, X_selected[:, 1].max() + 1, 100)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm.decision_function(xy).reshape(XX.shape)

# Contour plot for the decision boundary and margins
plt.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=1.0, linestyles=['--', '-', '--'], linewidths=3)

# Add labels and legend without bold styling
#plt.title(f"Accuracy ({accuracy * 100:.2f}%)", fontsize=32)
#plt.title("Accuracy", fontsize=32)
#plt.title("Accuracy")
#plt.xlabel(f"Most Effective Feature {most_effective_features[0] + 1}", fontsize=28)
#plt.ylabel(f"Most Effective Feature {most_effective_features[1] + 1}", fontsize=28)
#plt.legend(fontsize=24, loc='upper left', frameon=True)
# Remove x and y ticks
plt.xticks([])
plt.yticks([])
# Show the plot
results_path = os.path.join(results_dir, 'final_model.png')
plt.savefig(results_path, bbox_inches='tight', dpi=300)
plt.tight_layout()
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_curve, auc

# Apply Seaborn poster style
sns.set_context("poster")  # Poster style from Seaborn (larger fonts and markers)

# Generate synthetic data with good class separability to achieve AUC ~ 0.9
X, y = make_classification(
    n_samples=5000, n_features=10, n_informative=6, n_redundant=0,
    n_clusters_per_class=1, class_sep=1.8, flip_y=0.02, random_state=42
)

# Select the 2 most effective features using ANOVA F-statistic
selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X, y)
most_effective_features = selector.get_support(indices=True)

# Train an SVM classifier with probability estimates enabled
svm = SVC(kernel='linear', C=1, probability=True)  # Ensure probability=True is set
svm.fit(X_selected, y)

# Get predicted probabilities for ROC curve
y_prob = svm.predict_proba(X_selected)[:, 1]  # Probabilities for the positive class

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

# Create the ROC curve plot
plt.figure(figsize=(4, 4))  # Adjust the figure size
plt.plot(fpr, tpr, color='tab:blue', lw=2)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line (no discrimination)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('AUC')
# Remove x and y ticks
plt.xticks([])
plt.yticks([])
# Save the ROC curve plot
results_path = os.path.join(results_dir, 'roc_curve_auc.png')
plt.savefig(results_path, bbox_inches='tight', dpi=300)

# Show the plot
plt.tight_layout()
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score

# Apply Seaborn style
sns.set(style="whitegrid", context="poster")
#sns.set(context="poster")
# Generate a synthetic dataset
X, y = make_classification(
    n_samples=500, n_features=10, n_informative=6,
    n_redundant=0, random_state=42, class_sep=1.5
)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define kernels to evaluate
kernels = ['linear', 'rbf', 'poly', 'sigmoid']

# Initialize lists to store results
mean_accuracies = []
std_accuracies = []

# Evaluate each kernel
for kernel in kernels:
    svm = SVC(kernel=kernel, C=1, random_state=42)
    # Perform 5-fold cross-validation
    scores = cross_val_score(svm, X_train, y_train, cv=5, scoring='accuracy')
    mean_accuracies.append(scores.mean())
    std_accuracies.append(scores.std())

# Plot the results
plt.figure(figsize=(3, 3))
plt.bar(kernels, mean_accuracies, yerr=std_accuracies, capsize=5, color='gray')
plt.xlabel('Kernel Type')
plt.ylabel('Mean Accuracy')
#plt.title('Effect of Kernel on SVM Accuracy')
plt.ylim(0.5, 1.0)  # Set y-axis range for better visualization
plt.xticks([])
plt.yticks([])
# Customize spines to be black
# Get current axes
ax = plt.gca()

# Customize spines to set their color to black
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Set spine color to black
    spine.set_linewidth(2)
results_path = os.path.join(results_dir, 'tuning.png')
plt.savefig(results_path, bbox_inches='tight', dpi=300)
plt.tight_layout()
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Apply Seaborn style
sns.set(style="whitegrid", context="poster")

# Generate a synthetic dataset with 6 features
X, y = make_classification(
    n_samples=200, n_features=4, n_informative=4,
    n_redundant=0, random_state=42, class_sep=1.5
)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Generate all combinations of features from 1 to 6 features
feature_indices = list(range(X.shape[1]))
all_combinations = []

# Generate combinations of 1, 2, 3, 4, 5, and 6 features
for r in range(1, len(feature_indices) + 1):
    all_combinations.extend(combinations(feature_indices, r))

# Store results
results = []

# Evaluate accuracy for each combination
for feature_subset in all_combinations:
    # Select the corresponding feature subset
    X_train_subset = X_train[:, feature_subset]
    X_test_subset = X_test[:, feature_subset]

    # Train an SVM model
    svm = SVC(kernel='rbf', C=1, random_state=42)
    svm.fit(X_train_subset, y_train)

    # Compute the accuracy on the test set
    accuracy = svm.score(X_test_subset, y_test)
    results.append((feature_subset, accuracy))

# Sort results by accuracy
results = sorted(results, key=lambda x: x[1], reverse=True)

# Prepare data for visualization
feature_labels = [' + '.join([f'Feature {i}' for i in subset]) for subset, _ in results]
accuracies = [acc for _, acc in results]

# Plot the results
plt.figure(figsize=(4, 4))
sns.barplot(x=accuracies, y=feature_labels, palette="viridis")
plt.xlabel('Accuracy')
plt.ylabel('Feature Subsets')
#plt.title('Accuracy for All Feature Subsets from 1 to 6 Features')

# Adjust the y-axis tick labels and size
plt.yticks(fontsize=8)  # Adjust the font size of the y-tick labels
plt.xticks([])
plt.yticks([])
results_path = os.path.join(results_dir, 'featureSelection.png')
plt.savefig(results_path, bbox_inches='tight', dpi=300)
plt.tight_layout()
plt.show()

#%% ROIs
import numpy as np
from nilearn import plotting, image
from nilearn.image import math_img
from matplotlib import colors
parcellation_dir = '/Users/xiaoqianxiao/tool/parcellation'
Yeo_path = os.path.join(parcellation_dir, 'Yeo2011_17Networks_MNI152_FreeSurferConformed1mm.nii.gz')
#%%
network_labels = {
    "CEN": [9, 10],  # Example indices for CEN
    "DMN": [7, 8],   # Example indices for DMN
    "SA": [11, 12],  # Example indices for SA
    "LN": [15, 16],  # Example indices for LN
}

# Assign unique values to each network
network_values = {
    "CEN": 1,
    "DMN": 2,
    "SA": 3,
    "LN": 4,
}

# Create an empty image for combined networks
combined_mask = math_img("img * 0", img=yeo_17_path)

# Add each network to the combined mask
for network, indices in network_labels.items():
    mask_expr = " + ".join([f"(img == {i})" for i in indices])
    network_mask = math_img(mask_expr, img=yeo_17_path)
    combined_mask = math_img(f"img1 + ({network_values[network]} * img2)", img1=combined_mask, img2=network_mask)

# Define a colormap for the networks
cmap = colors.ListedColormap(["red", "blue", "green", "orange"])  # Colors for CEN, DMN, SA, LN
bounds = [0.5, 1.5, 2.5, 3.5, 4.5]  # Boundaries for the colormap
norm = colors.BoundaryNorm(bounds, cmap.N)
#%%
# Plot the combined networks
display = plotting.plot_roi(
    combined_mask,
    title="CEN, DMN, SA, and LN Networks",
    display_mode="ortho",
    draw_cross=False,
    cmap=cmap,
    colorbar=True
)

# Save the plot as an image
plt.savefig("combined_networks.png", dpi=300)
plt.show()  # Explicitly show the plot



#%%
from nilearn import plotting, image, datasets
import numpy as np
from matplotlib import colors
#%%
# Step 1: Fetch the Schaefer 2018 atlas with 400 ROIs and 17 networks
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=17)
atlas_img = image.load_img(atlas.maps)  # NIfTI image for the atlas
labels = atlas.labels   # Labels for each ROI
#%%
# Step 2: Decode byte strings to normal strings (if necessary)
decoded_labels = [label.decode("utf-8") if isinstance(label, bytes) else label for label in labels]
#%%
# Step 3: Identify indices for the sub-networks
# Create a new label array, initialize with zeros
new_label_data = np.zeros(atlas_img.shape)

# Assign labels for different sub-networks
# CEN (1-3) -> Red shades
conta_indices = [i + 1 for i, label in enumerate(decoded_labels) if "ContA" in label]
contb_indices = [i + 1 for i, label in enumerate(decoded_labels) if "ContB" in label]
contc_indices = [i + 1 for i, label in enumerate(decoded_labels) if "ContC" in label]
new_label_data[np.isin(atlas_img.get_fdata(), conta_indices)] = 1  # Red, lightness 1
new_label_data[np.isin(atlas_img.get_fdata(), contb_indices)] = 2  # Red, lightness 2
new_label_data[np.isin(atlas_img.get_fdata(), contc_indices)] = 3  # Red, lightness 3

# DMN (4-6) -> Green shades
defaulta_indices = [i + 1 for i, label in enumerate(decoded_labels) if "DefaultA" in label]
defaultb_indices = [i + 1 for i, label in enumerate(decoded_labels) if "DefaultB" in label]
defaultc_indices = [i + 1 for i, label in enumerate(decoded_labels) if "DefaultC" in label]
new_label_data[np.isin(atlas_img.get_fdata(), defaulta_indices)] = 4  # Green, lightness 4
new_label_data[np.isin(atlas_img.get_fdata(), defaultb_indices)] = 5  # Green, lightness 5
new_label_data[np.isin(atlas_img.get_fdata(), defaultc_indices)] = 6  # Green, lightness 6

# SA (7-8) -> Yellow shades
salventattna_indices = [i + 1 for i, label in enumerate(decoded_labels) if "SalVentAttnA" in label]
salventattnb_indices = [i + 1 for i, label in enumerate(decoded_labels) if "SalVentAttnB" in label]
new_label_data[np.isin(atlas_img.get_fdata(), salventattna_indices)] = 7  # Yellow, lightness 7
new_label_data[np.isin(atlas_img.get_fdata(), salventattnb_indices)] = 8  # Yellow, lightness 8

# LN (9-10) -> Blue shades
limbica_indices = [i + 1 for i, label in enumerate(decoded_labels) if "LimbicA" in label]
limbicb_indices = [i + 1 for i, label in enumerate(decoded_labels) if "LimbicB" in label]
new_label_data[np.isin(atlas_img.get_fdata(), limbica_indices)] = 9  # Blue, lightness 9
new_label_data[np.isin(atlas_img.get_fdata(), limbicb_indices)] = 10  # Blue, lightness 10
#%%
# Step 5: Create a new NIfTI image with the new label data
new_label_img = image.new_img_like(atlas_img, new_label_data)

# Step 6: Plot the new labels with assigned colors
# Assign color maps to the 10 new labels
#%%
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
roi_labels = ['ContA', 'ContB', 'ContC', 'DefaultA', 'DefaultB', 'DefaultC', 'SalVentAttnA', 'SalVentAttnB', 'LimbicA', 'LimbicB']
cmap = colormaps.get_cmap('tab20c')  # New colormap interface
color_indices = [0, 1, 2, 4, 5, 6, 8, 10, 12, 14]
colors = [cmap(i / 20) for i in color_indices]
custom_cmap = ListedColormap(colors)
#%%
# Plot the new label image
plt.clf()  # Clear the current figure
plt.close()  # Close the figure window
display = plotting.plot_roi(new_label_img, display_mode="ortho", draw_cross=False, cmap=custom_cmap)

# Step 7: Show the plot
#plotting.show()
# Add a custom legend
# Create two sets of handles: one for each row
handles = [plt.Line2D([0], [0], color=color, lw=4, markersize=10) for color in colors]
handles_row_1 = [handles[i] for i in [0, 1, 2, 6, 7]]  # First 5 handles (for first row)
handles_row_2 = [handles[i] for i in [3, 4, 5, 8, 9]]  # Last 5 handles (for second row)
labels_row_1 = [roi_labels[i] for i in [0, 1, 2, 6, 7]]
labels_row_2 = [roi_labels[i] for i in [3, 4, 5, 8, 9]]
# Create the first row of the legend
plt.legend(handles=handles_row_1, labels=labels_row_1, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=5, handlelength=1)
# Create the second row of the legend
#plt.legend(handles=handles_row_2, labels=labels_row_2, loc='upper center', bbox_to_anchor=(0.5, -0.45), ncol=5, handlelength=1)

# Adjust spacing to make room for the legend
#plt.subplots_adjust(bottom=2)


results_path = os.path.join(results_dir, 'ROIs.png')
plt.savefig(results_path, dpi=300, bbox_inches='tight')
# Show the plot
plt.show()
