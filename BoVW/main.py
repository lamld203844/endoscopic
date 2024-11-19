import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

from BoVW import BagOfVisualWords
# ===========================================================
# Initialization
# ===========================================================
np.random.seed(0)  # reproducibility
root_dir = '/teamspace/studios/this_studio/dataset/labeled-images'
ckpt_path = './checkpoints'
csv_result_path = f'{ckpt_path}/result.csv'

# os.makedirs(ckpt_path, exist_ok=True)
# os.makedirs(f'{ckpt_path}/descriptors/', exist_ok=True)
# os.makedirs(f'{ckpt_path}/codebook/', exist_ok=True)
# os.makedirs(f'{ckpt_path}/embedding_df/', exist_ok=True)
# os.makedirs(f'{ckpt_path}/best_model/', exist_ok=True)

s_id = np.random.randint(0, 100)

# ===========================================================
# HYPERPARAMs SWEEPING
# ===========================================================
'''hyperparameters'''
hyperparams = {
    'feature_extractor': 'SIFT',
    'strongest_percent': 1,
    'clustering_algorithm': 'KMeans',
    'vector_size': 128
}
# descriptors extracting
method = hyperparams['feature_extractor']
strongest_percent = hyperparams['strongest_percent']
# quantization
cluster_algorithm = hyperparams['clustering_algorithm']
k = hyperparams['vector_size'] # #cluster = vector size
'''hyperparameters'''

# --------------------------------------------------------------
# ----- 1. extracting descriptors -------------
# --------------------------------------------------------------
model = BagOfVisualWords(
        root_dir=root_dir,
        method=method
    )
# sample_size = len(model.df)
sample_size = 20
descriptors_lake = model.extract_descriptors(sample_size=sample_size,
                                            strongest_percent=strongest_percent)

# saving 
descriptors_lake_path = joblib.dump(descriptors_lake,
                                    f'{ckpt_path}/descriptors/id{s_id}-{sample_size}_img-{model.method}_extractor-{strongest_percent*100}%_strongest.pkl',
                                    compress=3)
del descriptors_lake # free memory
print('='*20, 'Completely extracting descriptors', '='*20)

# =================== load ================================
descriptors_lake = joblib.load(*descriptors_lake_path) # unpack list


# --------------------------------------------------------------
#  2. building codebook (visual vocabulary) 
# --------------------------------------------------------------
codebook = model.build_codebook(descriptors_lake=descriptors_lake,
                                    cluster_algorithm='kmean',
                                    k=200
                                )
# saving 
codebook_path = joblib.dump(codebook,
                            f'{ckpt_path}/codebook/id{s_id}-{cluster_algorithm}_cluster_algorithm-k={k}.pkl',
                            compress=3)
del descriptors_lake # free memory
del codebook # free memory
print('='*20, 'Completely building codebook', '='*20)

# --------------------------------------------------------------
# 3. Embedding representation 
# --------------------------------------------------------------

codebook = joblib.load(*codebook_path) # unpack list
# n_imgs = len(model.df)
n_imgs = 50 # for test

headers = [f'feature{i}' for i in range(codebook.shape[0])]
embedding_df = pd.DataFrame(columns=headers)
labels = []

# Embedding entire dataset
normalized = False
try:
    for idx in tqdm(range(n_imgs)):
        img, label = model._get_item(idx)
        embedding = model.get_embedding(idx, codebook, normalized=normalized)
        # Add a row to the DataFrame
        embedding_df.loc[len(embedding_df)] = embedding
        labels.append(label)
        # break
except Exception as e:
    print(e)
    pass

embedding_df['label'] = pd.Series(labels, dtype='int')

# saving
embedding_df_path = f'{ckpt_path}/embedding_df/id{s_id}-{n_imgs}_img-normalized={normalized}.csv'
embedding_df.to_csv(embedding_df_path, index=False)

del embedding_df # free memory
print('='*20, 'Completely building embedding representation', '='*20)


# --------------------------------------------------------------
# 4. Classification 
# --------------------------------------------------------------
df = pd.read_csv(embedding_df_path)

X = df.drop('label', axis=1)
y = df['label']

from utils import TPOT_autoML, evaluate_model

metadata = os.path.basename(embedding_df_path)
best_model = TPOT_autoML(X, y, ckpt_path, metadata)
# Evaluate the best model using custom metrics
metrics = evaluate_model(X, y, best_model)

## ==================== Accumulating saving final result ======================
results = []
results.append({**hyperparams, **metrics})
# Convert results to a DataFrame and save to CSV
results_df = pd.DataFrame(results)

# If file exists, append; otherwise, create with headers
if csv_result_path:
    try:
        existing_df = pd.read_csv(csv_result_path)
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
    except FileNotFoundError:
        pass

results_df.to_csv(csv_result_path, index=False)