import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import export_text, plot_tree

# Try SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False


# =====================================================
# CONFIGURATION
# =====================================================

MODEL_PATH = "rf_credit_risk_model.pkl"
DATA_PATH = "X_test_data.csv"
OUTPUT_DIR = "rf_interpretability_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# LOAD TEST DATA
# =====================================================

print("Loading test data...")

X_test = pd.read_csv(DATA_PATH, index_col=0)

feature_names = list(X_test.columns)

print(f"Features detected: {len(feature_names)}")


# =====================================================
# LOAD MODEL WITH JOBLIB (FIXED)
# =====================================================

print("\nLoading model using joblib...")

model = joblib.load(MODEL_PATH)

print("Loaded object type:", type(model))


# =====================================================
# HANDLE PIPELINE OR DIRECT MODEL
# =====================================================

rf = None

# Case 1: Direct RandomForest
if "RandomForest" in str(type(model)):
    
    rf = model
    print("Direct RandomForest detected")


# Case 2: Pipeline
elif hasattr(model, "named_steps"):
    
    print("Pipeline detected. Searching for RandomForest...")

    for name, step in model.named_steps.items():

        if "RandomForest" in str(type(step)):

            rf = step
            print(f"RandomForest found in pipeline step: {name}")
            break


# Error if not found
if rf is None:

    raise Exception("RandomForest model not found inside file.")


# =====================================================
# FEATURE IMPORTANCE
# =====================================================

print("\nGenerating feature importance...")

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": rf.feature_importances_
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)


print("\nTop 10 most important features:")

print(importance_df.head(10))


# Save CSV
importance_df.to_csv(
    f"{OUTPUT_DIR}/feature_importance.csv",
    index=False
)


# Plot
plt.figure(figsize=(10,6))

plt.barh(
    importance_df["feature"][:15],
    importance_df["importance"][:15]
)

plt.gca().invert_yaxis()

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/feature_importance.png"
)

plt.show()


# =====================================================
# EXTRACT TREE RULES
# =====================================================

print("\nExtracting tree rules...")

tree = rf.estimators_[0]

rules = export_text(
    tree,
    feature_names=feature_names
)


# Save rules
with open(
    f"{OUTPUT_DIR}/tree_rules.txt",
    "w"
) as f:

    f.write(rules)


print("\nTree rules preview:\n")

print(rules[:1000])


# =====================================================
# TREE VISUALIZATION
# =====================================================

print("\nGenerating tree visualization...")

plt.figure(figsize=(20,10))

plot_tree(
    tree,
    feature_names=feature_names,
    filled=True,
    max_depth=3,
    fontsize=10
)

plt.title("Decision Tree Visualization (depth limited)")

plt.savefig(
    f"{OUTPUT_DIR}/tree_visualization.png"
)

plt.show()


# =====================================================
# SHAP ANALYSIS
# =====================================================

if SHAP_AVAILABLE:

    print("\nRunning SHAP analysis...")

    explainer = shap.TreeExplainer(rf)

    shap_values = explainer.shap_values(X_test)

    # FIX: use class 1 (default class)
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values

    plt.figure()

    shap.summary_plot(
        shap_values_to_plot,
        X_test,
        plot_type="bar",
        show=False
    )

    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/shap_summary.png", dpi=300)

    plt.show()

else:

    print("\nSHAP not installed.")
    print("Install using: pip install shap")


# =====================================================
# DONE
# =====================================================

print("\nInterpretability analysis complete.")

print(f"\nAll outputs saved in folder: {OUTPUT_DIR}")