import os
import glob

import torch
import numpy as np
import joblib


class LinearProbe(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


SEQUENCE_LENGTH = 5000
INPUT_DIM = SEQUENCE_LENGTH * 4
SCALER_PATH = "scaler.joblib"
MODEL_PATH = "best_linear_probe.pt"


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


def normalize_sequence_length(seq, target_len):
    seq_len = len(seq)

    if seq_len == target_len:
        return seq

    if seq_len > target_len:
        start = (seq_len - target_len) // 2
        return seq[start:start + target_len]

    pad_total = target_len - seq_len
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    return "N" * pad_left + seq + "N" * pad_right


def one_hot_encode(seq):
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    encoding = np.zeros((len(seq), 4), dtype=np.float32)

    for i, base in enumerate(seq):
        idx = mapping.get(base)
        if idx is not None:
            encoding[i, idx] = 1.0

    return encoding.flatten()


def predict_fasta(fasta_path, model, scaler):
    records = read_fasta(fasta_path)

    if not records:
        return []

    embeddings = np.vstack([
        one_hot_encode(normalize_sequence_length(seq, SEQUENCE_LENGTH))
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
    scaler = joblib.load(SCALER_PATH)

    model = LinearProbe(INPUT_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    fasta_files = sorted(glob.glob("*.fa"))

    if not fasta_files:
        raise FileNotFoundError("No .fa files found in the current directory.")

    for fasta_path in fasta_files:
        predictions = predict_fasta(fasta_path, model, scaler)

        base_name = os.path.splitext(os.path.basename(fasta_path))[0]
        output_path = f"{base_name}_onehot_probabilities.tsv"

        with open(output_path, "w") as f:
            f.write("sequence_id\tprobability\n")
            for name, prob in predictions:
                f.write(f"{name}\t{prob:.6f}\n")

        print(f"{fasta_path} -> {output_path}")


if __name__ == "__main__":
    main()