# SoundEmbeddings

This small package conveniently embeds valid IPA sounds and represents them in 128 dimensions. Embeddings are trained on lexical data from the [lexibank-analysed](https://github.com/lexibank/lexibank-analysed) dataset, using a relatively vanilla Word2Vec SkipGram model.

## Installation

This package can be conveniently installed via pip.

```
pip install soundembeddings
```

## Usage

### Accessing CLTS

The package makes use of the [CLTS](https://clts.clld.org) catalog under the hood. You need a clone of the CLTS repository on your system:

```
git clone --depth 1 https://github.com/cldf-clts/clts
```

If you already have an installation of CLTS (e.g. via `cldfbench`), this step can be omitted.

### Retrieving Embeddings

Once you are set up, embeddings can be retrieved from the `SoundEmbeddings` object.

```python
se = SoundEmbeddings(PATH_TO_CLTS_REPOS)  # initialize the object
example_embedding = se("tʃ")  # get an embedding for an arbitrary IPA symbol
similarity = se.similarity("tʃ", "ts")  # calculate the cosine similarity between two sound embeddings (in this case, ~0.878)
```

If you have a running [CLDFBench](https://github.com/cldf/cldfbench) configuration with a default CLTS repository, the path can be omitted.

```python
se = SoundEmbeddings()  # works only if a default CLTS repo is set up via 'cldfbench catconfig'
```

## Replicating the training workflow

Replicating the training routine requires the installation of PyTorch and the cloning of the `lexibank-analysed` dataset, as well of this repository, since the training routine is not packaged. After this, the `train.py` can be run and adjusted ad libitum.

```
git clone https://github.com/arubehn/soundembeddings
cd soundembeddings
pip install .[train]
git clone --depth 1 https://github.com/lexibank/lexibank-analysed/
python train.py
```

Note that this training routine is non-deterministic, so results are expected to deviate somewhat.
