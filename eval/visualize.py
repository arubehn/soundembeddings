import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from soundembeddings import SoundEmbeddings
from seaborn import heatmap
from pathlib import Path

# the 25 most common consonants and 20 most common vowels, according to phoible
core_consonants = ["p", "b", "m", "w", "f", "v", "t", "t̪", "d", "s", "z", "r", "ɾ", "l",
                   "n", "ʃ", "tʃ", "dʒ", "j", "ɲ", "k", "g", "ŋ", "ʔ", "h"]
core_vowels = ["i", "iː", "ĩ", "ɪ", "e", "eː", "ɛ", "a", "aː", "ã",
               "ɨ", "ə", "u", "uː", "ũ", "ʊ", "o", "oː", "õ", "ɔ"]
german_sounds = ["m", "j", "a", "n", "l", "s", "b", "ŋ", "g", "h", "f", "tʃ", "ɛ", "ʃ", "ɔ", "iː", "aː",
                 "z", "uː", "dʒ", "v", "ə", "ts", "eː", "oː", "kʰ", "pʰ", "ʒ", "ɪ", "ʊ", "ɛː", "ç", "œ",
                 "yː", "øː", "pf", "ʏ", "ʀ", "tʰ", "d"]


BASE_DIR = Path(__file__).parent


def plot(func):
    def inner(*args):
        plt.cla()
        plt.clf()
        func(*args)
    return inner


@plot
#def cos_similarity_heatmap(vectors, sounds, name):
#    sim_matrix = cosine_similarity(vectors)
#    heatmap(sim_matrix, xticklabels=sounds, yticklabels=sounds)
#    plt.savefig(name)


@plot
def plot_pca(vectors, sounds, name):
    pca = PCA(n_components=2)
    res = pca.fit_transform(vectors)
    plt.scatter(*np.swapaxes(res, 0, 1))
    for sound, coordinates in zip(sounds, res):
        plt.annotate(sound, coordinates)
    plt.savefig(name)


@plot
def plot_tsne(vectors, sounds, name):
    tsne = TSNE(n_components=2, perplexity=5)
    res = tsne.fit_transform(np.array(vectors))
    plt.scatter(*np.swapaxes(res, 0, 1))
    for sound, coordinates in zip(sounds, res):
        plt.annotate(sound, coordinates)
    plt.savefig(name)


@plot
def sim_heatmap(embeddings, sounds, name):
    matrix = np.zeros((len(sounds), len(sounds)))
    for i, sound1 in enumerate(sounds):
        for j, sound2 in enumerate(sounds[i:], i):
            matrix[i, j] = matrix[j, i] = embeddings.similarity(sound1, sound2)
    heatmap(matrix, xticklabels=sounds, yticklabels=sounds)
    plt.savefig(name)


if __name__ == "__main__":
    embeddings = SoundEmbeddings()
    for sample, name in [
            (core_vowels, "corev"),
            (core_consonants, "corec"),
            (core_vowels + core_consonants, "corecv"),
            (german_sounds, "german")]:
        vecs = [embeddings(x) for x in sample]
        sim_heatmap(embeddings, sample, BASE_DIR / f"{name}-heatmap.pdf")
        plot_pca(vecs, sample, BASE_DIR / f"{name}-pca.pdf")
        plot_tsne(vecs, sample, BASE_DIR / f"{name}-tsne.pdf")
