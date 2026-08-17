import numpy as np
from importlib import resources
from soundvectors import SoundVectors
from pyclts import CLTS


class SoundEmbeddings(object):
    def __init__(self, clts_repos=None):
        bipa = CLTS(clts_repos).bipa
        self.sv = SoundVectors(ts=bipa)
        self.weights = np.transpose(self._load_weights())

    def _load_weights(self):
        with resources.files("soundembeddings.data").joinpath("weights.npy").open("rb") as f:
            return np.load(f)

    def embed(self, s):
        vec = self.sv.get_vec(s)
        return np.matmul(vec, self.weights)

    def __call__(self, s):
        return self.embed(s)

    def similarity(self, s1, s2):
        emb1 = self.embed(s1)
        emb2 = self.embed(s2)
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
