import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pycldf import Dataset
from pathlib import Path
from soundembeddings import SoundEmbeddings
from itertools import combinations


INDEX_PATH = Path(__file__).parent.parent.parent / "indexdiachronica" / "cldf" / "StructureDataset-metadata.json"
index_diachronica = Dataset.from_metadata(INDEX_PATH)

feat_vec_sims = []
emb_sims = []

se = SoundEmbeddings()
sounds = set()

for row in index_diachronica.iter_rows("ParameterTable"):
    source = row["Source"]
    target = row["Target"]
    count = row["Count"]
    if count < 2:
        continue
    try:
        source_vec = se.sv.get_vec(source)
        target_vec = se.sv.get_vec(target)
        vector_similarity = float(np.dot(source_vec, target_vec) / (np.linalg.norm(source_vec) * np.linalg.norm(target_vec)))
        embedding_similarity = se.similarity(source, target)
        for _ in range(int(np.floor(np.log2(count)))):
            feat_vec_sims.append(vector_similarity)
            emb_sims.append(embedding_similarity)
        sounds.add(source)
        sounds.add(target)
    except ValueError as ve:
        print(ve, source, target)

random_feat_vec_sims = []
random_emb_sims = []

for sound1, sound2 in combinations(sounds, 2):
    source_vec = se.sv.get_vec(sound1)
    target_vec = se.sv.get_vec(sound2)
    vector_similarity = float(
        np.dot(source_vec, target_vec) / (np.linalg.norm(source_vec) * np.linalg.norm(target_vec)))
    embedding_similarity = se.similarity(sound1, sound2)
    random_feat_vec_sims.append(vector_similarity)
    random_emb_sims.append(embedding_similarity)

fig, ax = plt.subplots(figsize=(8, 5))

sns.kdeplot(feat_vec_sims, fill=True, alpha=0.3, clip=(-1, 1), label="Feature Vectors", ax=ax)
sns.kdeplot(emb_sims, fill=True, alpha=0.3, clip=(-1, 1), label="Embeddings", ax=ax)
sns.kdeplot(random_feat_vec_sims, fill=True, alpha=0.3, clip=(-1, 1), label="Feature Vectors (Random)", ax=ax)
sns.kdeplot(random_emb_sims, fill=True, alpha=0.3, clip=(-1, 1), label="Embeddings (Random)", ax=ax)


ax.set_xlim(-1, 1)
ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
plt.show()
