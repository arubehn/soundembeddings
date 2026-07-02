import torch
import json
from soundvectors import SoundVectors
from pycldf import Dataset
from pyclts import CLTS
from collections import Counter
from tqdm import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IGNORE_SEGMENTS = ["+", "∼"]


class SkipGram(torch.nn.Module):
    """
    Implementation of Skip-Gram model described in paper:
    https://arxiv.org/abs/1301.3781

    Taken from https://github.com/OlgaChernytska/word2vec-pytorch/blob/main/utils/model.py (with slight modifications)
    """
    def __init__(self, vocab_size: int, input_dim: int, embed_dimension: int = 128):
        super(SkipGram, self).__init__()
        self.embeddings = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=input_dim,
                out_features=embed_dimension,
                bias=False,
            )
        )
        self.linear = torch.nn.Linear(
            in_features=embed_dimension,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = self.linear(x)
        return x


class NCELoss(torch.nn.Module):
    def __init__(self, **kwargs):
        super(NCELoss, self).__init__(**kwargs)

    def forward(self, pred, y):
        """
        Defines the loss function under negative sampling after Mikolov et al. (2013).
        :param y: the "true" labels with negative sampling. the actual node should be 1, the randomly sampled distractors -1, everything else 0.
        :param pred: the logits, where each index corresponds to a node.
        :return: the loss of the batch
        """
        return -torch.sum(torch.log(torch.sigmoid(y * pred))) / pred.size(0)


class SoundContextDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class Sound2Vec(object):
    def __init__(self, sv: SoundVectors):
        self.sv = sv
        self.device = DEVICE
        self.sound2id = {}
        self.id2sound = {}
        self.sound_counts = Counter()
        self.embeddings = None
        self.embedding_layer = None

    def generate_training_data(self, forms, window_size=1):
        target_sounds, context_bow = [], []

        for form in tqdm(forms, desc="Generating training data (1)"):
            for i, sound in enumerate(form):
                left_idx = max(0, i - window_size)
                right_idx = min(len(form), i + window_size + 1)
                context = form[left_idx:i] + form[i + 1:right_idx]
                target_sounds.append(sound)
                context_bow.append(context)

        X, Y = [], []
        for sound_id, context in tqdm(zip(target_sounds, context_bow), desc="Generating training data (2)", total=len(target_sounds)):
            sound = self.id2sound.get(sound_id)
            if not sound:
                continue
            for context_sound_id in context:
                context_sound = self.id2sound.get(context_sound_id)
                if not context_sound:
                    continue
                try:
                    X.append(self.sv.get_vec(sound))
                    Y.append(context_sound_id)
                except ValueError:
                    print(f"{sound} could not be encoded!")

        return SoundContextDataset(X, Y)

    def tokenize_forms(self, forms, freq_threshold=10):
        for form in tqdm(forms, desc="Counting forms"):
            self.sound_counts.update(form)

        id = 0
        for sound, count in self.sound_counts.items():
            if sound not in IGNORE_SEGMENTS and count >= freq_threshold:
                self.id2sound[id] = sound
                self.sound2id[sound] = id
                id += 1

        tokenized_forms = []

        for form in tqdm(forms, desc="Tokenizing forms"):
            tokenized_form = []
            for segment in form:
                if segment in IGNORE_SEGMENTS:
                    continue
                id = self.sound2id.get(segment, -1)
                tokenized_form.append(id)
            tokenized_forms.append(tokenized_form)

        return tokenized_forms

    def train(self, forms, freq_threshold=10, **kwargs):
        tokenized_forms = self.tokenize_forms(forms, freq_threshold=freq_threshold)
        dataset = self.generate_training_data(tokenized_forms, **kwargs)
        test_split = kwargs.get("test_split", 0.2)
        train_set, test_set = torch.utils.data.random_split(dataset, [1-test_split, test_split])
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=kwargs.get("batch_size", 1028), shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=kwargs.get("batch_size", 1028), shuffle=True)

        model = SkipGram(len(self.sound2id), dataset.X.shape[1])
        model.to(self.device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=kwargs.get("lr", 0.001))

        best_loss = torch.inf
        wait = 0
        training_progress = tqdm(
            range(kwargs.get("max_epochs", 5000)),
            desc=f"Training Sound2Vec...",
        )

        for epoch in training_progress:
            model.train()
            epoch_loss = 0
            for X_train, Y_train in train_dataloader:
                X_train.to(self.device)
                Y_train.to(self.device)
                pred = model(X_train)
                loss = criterion(pred, Y_train)
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0
            for X_test, Y_test in test_dataloader:
                with torch.no_grad():
                    val_loss += criterion(model(X_test), Y_test).item()

            training_progress.set_description(
                f"Training Sound2Vec | "
                f"Loss (train): {float(epoch_loss):.4f} | Loss (val): {float(val_loss):.4f}"
            )

            if val_loss - best_loss < -kwargs.get("min_delta", 0):
                best_loss = val_loss
                wait = 0
            else:
                wait += 1
                if kwargs.get("patience", 5) is not None and wait > kwargs.get("patience", 5):
                    print(f"Training stopped after {epoch} epochs.")
                    break

        model.eval()
        with torch.no_grad():
            self.embeddings = {
                sound: model.embeddings(
                    torch.tensor(self.sv.get_vec(sound), device=self.device).to(torch.float32)
                ).detach().cpu().tolist()
                for i, sound in self.id2sound.items() if self.sv.validate(sound)
            }
            self.embedding_layer = model.embeddings


if __name__ == "__main__":
    ds = Dataset.from_metadata("lexibank-analysed/cldf/wordlist-metadata.json")
    forms = []
    for form in tqdm(ds.iter_rows("FormTable"), desc="Collecting forms from Lexibank"):
        forms.append(form["Segments"])
    bipa = CLTS().bipa
    sv = SoundVectors(ts=bipa)

    sound2vec = Sound2Vec(sv)
    sound2vec.train(forms)
    torch.save(sound2vec.embedding_layer, "embeddings.pt")
    with open("embeddings.json", "w") as f:
        json.dump(sound2vec.embeddings, f)
