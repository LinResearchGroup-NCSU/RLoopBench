import os
import glob
import itertools
from collections import Counter

import torch
import numpy as np
import joblib


class LinearProbe(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


K = 4
SCALER_PATH = "scaler.joblib"
MODEL_PATH = "best_linear_probe.pt"
OUTPUT_DIR = "."
# OUTPUT_DIR = "predictions"


def generate_kmers(k):
    bases = ["A", "C", "G", "T"]
    return ["".join(p) for p in itertools.product(bases, repeat=k)]


def read_fasta(path):
    records = []
    name = None
    seq = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(seq).upper()))
                name = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)

        if name is not None:
            records.append((name, "".join(seq).upper()))

    return records


def get_kmer_embedding(sequence, k, kmer_to_idx):
    vector = np.zeros(len(kmer_to_idx), dtype=np.float32)

    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        idx = kmer_to_idx.get(kmer)
        if idx is not None:
            vector[idx] += 1

    total = vector.sum()
    if total > 0:
        vector /= total

    return vector


def predict_fasta(fasta_path, model, scaler, kmer_to_idx):
    records = read_fasta(fasta_path)

    if not records:
        return []

    embeddings = np.vstack([
        get_kmer_embedding(seq, K, kmer_to_idx)
        for _, seq in records
    ])

    embeddings = scaler.transform(embeddings)
    x = torch.tensor(embeddings, dtype=torch.float32)

    with torch.no_grad():
        probabilities = torch.sigmoid(model(x)).cpu().numpy().flatten()

    return [
        (name, prob)
        for (name, _), prob in zip(records, probabilities)
    ]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_kmers = generate_kmers(K)
    kmer_to_idx = {kmer: i for i, kmer in enumerate(all_kmers)}

    scaler = joblib.load(SCALER_PATH)

    model = LinearProbe(input_dim=len(all_kmers))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    fasta_files = sorted(glob.glob("*.fa"))

    if not fasta_files:
        raise FileNotFoundError("No .fa files found in the current directory.")

    for fasta_path in fasta_files:
        predictions = predict_fasta(fasta_path, model, scaler, kmer_to_idx)

        base_name = os.path.splitext(os.path.basename(fasta_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}_probabilities.tsv")

        with open(output_path, "w") as f:
            f.write("sequence_id\tprobability\n")
            for name, prob in predictions:
                f.write(f"{name}\t{prob:.6f}\n")

        print(f"{fasta_path} -> {output_path}")


if __name__ == "__main__":
    main()