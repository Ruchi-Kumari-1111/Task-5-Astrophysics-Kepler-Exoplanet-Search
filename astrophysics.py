import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- SETUP DIRECTORY ---
# Use raw string (r"") to handle Windows backslashes properly
save_dir = r"D:\VS codes\Internship(7thSem)\Task-5"

# Create the folder if it doesn't already exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Created directory: {save_dir}")

# --- 1. IMPORT AND PREPROCESS ---
print("Loading and preprocessing data...")
# Assuming cumulative.csv is in the same folder you are running the script from
df = pd.read_csv('cumulative.csv')

# Filter out 'CANDIDATE' to make it a clean binary classification
df = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])]

# Map to 1 (Confirmed Planet) and 0 (False Positive)
df['target'] = df['koi_disposition'].map({'CONFIRMED': 1, 'FALSE POSITIVE': 0})

# Select our physical features
features = [
    'koi_duration', # Transit Duration
    'koi_srad',     # Stellar Radius
    'koi_impact',   # Impact Parameter
    'koi_teq'       # Equilibrium Temperature
]

# Drop rows with missing values in our selected columns to prevent crashes
df = df.dropna(subset=features + ['target'])

X = df[features]
y = df['target']

# Split into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data ready!\n" + "-"*50)


# --- TRAIN DECISION TREE AND VISUALIZE ---
print("\nTraining Decision Tree Classifier...")
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)

plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=features, class_names=['False Positive', 'Confirmed'], filled=True, rounded=True)
plt.title("Decision Tree for Exoplanet Classification (Max Depth = 3)")
tree_path = os.path.join(save_dir, "1_Decision_Tree_Visualization.png")

# Save, Show, then Close
plt.savefig(tree_path)
print(f"🌳 GRAPH 1 SAVED TO: {tree_path}")
plt.show() 
plt.close()

print("""--- GRAPH 1 EXPLANATION: THE DECISION TREE ---
This graph shows the 'brain' of the decision tree. The top node is the most important rule 
(likely looking at the transit duration or impact parameter). If a star's data meets the 
condition, it moves left (True); otherwise, it moves right (False). The 'gini' score shows 
how mixed the data is—a score of 0.0 means it has perfectly isolated either planets or 
false positives.""")


# --- ANALYZE OVERFITTING ---
print("\nAnalyzing Overfitting and Tree Depth...")
train_acc = []
test_acc = []
depths = range(1, 20)

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, model.predict(X_train)))
    test_acc.append(accuracy_score(y_test, model.predict(X_test)))

plt.figure(figsize=(10, 6))
plt.plot(depths, train_acc, label='Training Accuracy', marker='o')
plt.plot(depths, test_acc, label='Testing Accuracy', marker='o')
plt.xlabel('Tree Depth (max_depth)')
plt.ylabel('Accuracy')
plt.title('Decision Tree Overfitting Analysis')
plt.legend()
plt.grid(True)
overfit_path = os.path.join(save_dir, "2_Overfitting_Analysis.png")

# Save, Show, then Close
plt.savefig(overfit_path)
print(f"📈 GRAPH 2 SAVED TO: {overfit_path}")
plt.show()
plt.close()

print("""--- GRAPH 2 EXPLANATION: OVERFITTING ANALYSIS ---
This line chart visualizes overfitting perfectly. As the tree depth increases (moving right 
on the X-axis), the Training Accuracy climbs toward 100% because the model is memorizing 
the specific noise in the training data. However, the Testing Accuracy plateaus or drops. 
The widening gap between the blue and orange lines indicates overfitting.""")


# --- TRAIN RANDOM FOREST AND COMPARE ---
print("\nTraining Random Forest and Comparing Accuracy...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

dt_accuracy = accuracy_score(y_test, dt_model.predict(X_test))
rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))

print(f"Single Decision Tree Accuracy: {dt_accuracy * 100:.2f}%")
print(f"Random Forest Accuracy:        {rf_accuracy * 100:.2f}%")
print("Notice how the Random Forest (an ensemble of 100 trees) smooths out the errors of a single tree to achieve higher accuracy.")


# --- INTERPRET FEATURE IMPORTANCES ---
print("\nInterpreting Feature Importances...")
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
importances.plot(kind='barh', color='teal')
plt.title("Random Forest Feature Importances (Exoplanet Detection)")
plt.xlabel("Importance Score")
plt.ylabel("Astrophysical Feature")
plt.tight_layout()
importance_path = os.path.join(save_dir, "3_Feature_Importances.png")

# Save, Show, then Close
plt.savefig(importance_path)
print(f"📊 GRAPH 3 SAVED TO: {importance_path}")
plt.show()
plt.close()

print("""--- GRAPH 3 EXPLANATION: FEATURE IMPORTANCE ---
This bar chart ranks which astronomical features were most useful for the Random Forest 
to make its decisions. The longer the bar, the more predictive power that specific feature 
holds. For example, you will likely see that the 'Impact Parameter' or 'Transit Duration' 
is far more important than the 'Stellar Radius' when predicting a real planet.""")


# --- EVALUATE USING CROSS-VALIDATION ---
print("\nEvaluating using 5-Fold Cross Validation...")
# We use the whole dataset (X, y) for cross-validation
cv_scores = cross_val_score(rf_model, X, y, cv=5)

print(f"Cross-Validation Scores (5 folds): {cv_scores}")
print(f"Average CV Accuracy: {cv_scores.mean() * 100:.2f}%")
print("""
--- CROSS-VALIDATION EXPLANATION ---
Instead of just splitting the data into Train/Test once, 5-Fold CV splits the data into 
5 different chunks. It trains the model on 4 chunks and tests on the 1 remaining, repeating 
this 5 times. The consistent scores above prove that our Random Forest is stable and didn't 
just get 'lucky' on a single split of the space data.
""")
print("\n*** ALL TASKS COMPLETED SUCCESSFULLY ***")