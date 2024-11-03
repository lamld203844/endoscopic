import pandas as pd
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, make_scorer, f1_score

# 1. Load dataset
df = pd.read_csv('/media/mountHDD2/lamluuduc/endoscopy/base-code/endoscopic/checkpoints/embedding_df/id44-10662_img-normalized=False.csv')

X = df.drop('label', axis=1)
y = df['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define a custom scoring function (e.g., F1 macro)
f1_macro_scorer = make_scorer(f1_score, average='macro')

# Set up TPOT with parallelism and scoring for F1 macro
tpot = TPOTClassifier(
    generations=5,             # Number of generations to run
    population_size=20,         # Number of pipelines to try per generation
    scoring=f1_macro_scorer,    # Optimize for F1 macro score
    cv=5,                       # 5-fold cross-validation
    n_jobs=-1,                  # Use all available CPU cores
    random_state=42,
    verbosity=2                 # Verbosity level for tracking progress
)

# Fit TPOT to the training data
tpot.fit(X_train, y_train)

# Make predictions on the test set and evaluate performance
y_pred = tpot.predict(X_test)
print("Test F1 Macro Score:", f1_score(y_test, y_pred, average='macro'))

# Export the best pipeline
tpot.export('best_model_pipeline.py')


