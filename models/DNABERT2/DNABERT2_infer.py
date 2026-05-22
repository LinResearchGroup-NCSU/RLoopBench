import os
import sys
import glob

import torch
import numpy as np
import pandas as pd
import joblib

from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig


MODEL_NAME = "zhihan1996/DNABERT-2-117M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WINDOW_SIZE = 5000
STEP_SIZE = 500

SCALER_PATH = "scaler.joblib"
PROBE_PATH = "best_linear_probe.pt"
PROBE_INPUT_DIM = 768


if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = np.core


class LinearProbe(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def read_fasta(file_path):
    headers = []
    sequences = []

    seq = ""
    header = None

    with open(file_path) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if seq:
                    headers.append(header)
                    sequences.append(seq)

                header = line[1:]
                seq = ""
            else:
                seq += line.upper()

        if seq:
            headers.append(header)
            sequences.append(seq)

    return headers, sequences


def split_sequence_windows(seq, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    seq_len = len(seq)

    if seq_len == 0:
        return [], []

    if seq_len <= window_size:
        return [seq], [0]

    windows = []
    starts = []

    for start in range(0, seq_len - window_size + 1, step_size):
        end = start + window_size
        windows.append(seq[start:end])
        starts.append(start)

    last_start = seq_len - window_size

    if starts[-1] != last_start:
        windows.append(seq[last_start:seq_len])
        starts.append(last_start)

    return windows, starts


def masked_mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)

    return summed / counts


print(f"Loading DNABERT-2: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
config = BertConfig.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    config=config
)

bert_model.eval().to(DEVICE)

print("Loading scaler and linear probe...")
scaler = joblib.load(SCALER_PATH)

probe = LinearProbe(PROBE_INPUT_DIM)
probe.load_state_dict(torch.load(PROBE_PATH, map_location=DEVICE))
probe.eval().to(DEVICE)


def get_score_for_sequence(dna):
    inputs = tokenizer(
        dna,
        return_tensors="pt",
        padding=False,
        truncation=False,
        return_attention_mask=True
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = bert_model(**inputs)
        pooled = masked_mean_pool(outputs[0], inputs["attention_mask"])

        emb_np = pooled.cpu().numpy()
        emb_scaled = scaler.transform(emb_np)

        x_tensor = torch.tensor(emb_scaled, dtype=torch.float32, device=DEVICE)
        prob = torch.sigmoid(probe(x_tensor)).item()

    return float(prob)


def process_fasta(fasta_file):
    print(f"\nProcessing {fasta_file}")

    headers, seqs = read_fasta(fasta_file)

    window_rows = []
    summary_rows = []

    for idx, (header, seq) in enumerate(zip(headers, seqs), start=1):
        if len(seq) == 0:
            continue

        if idx % 10 == 0:
            print(f"Processed {idx}/{len(seqs)} sequences")

        windows, starts = split_sequence_windows(seq)
        seq_scores = []

        for window_seq, start in zip(windows, starts):
            score = get_score_for_sequence(window_seq)
            end = start + len(window_seq)

            seq_scores.append(score)

            window_rows.append({
                "header": header,
                "sequence_length": len(seq),
                "window_start_0based": start,
                "window_end_0based_exclusive": end,
                "window_length": len(window_seq),
                "score": score
            })

        summary_rows.append({
            "header": header,
            "sequence_length": len(seq),
            "num_windows": len(seq_scores),
            "max_score": float(np.max(seq_scores)),
            "mean_score": float(np.mean(seq_scores))
        })

    base = os.path.splitext(os.path.basename(fasta_file))[0]

    summary_csv = f"{base}_summary.csv"
    window_csv = f"{base}_raw.csv"

    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    pd.DataFrame(window_rows).to_csv(window_csv, index=False)

    print(f"Done: {fasta_file}")
    print(f"Saved summary to: {summary_csv}")
    print(f"Saved window scores to: {window_csv}")


def main():
    fasta_files = sorted(glob.glob("*.fa"))

    if not fasta_files:
        raise FileNotFoundError("No .fa files found in the current directory.")

    print(f"Found {len(fasta_files)} FASTA files:")

    for fasta in fasta_files:
        print(f"  - {fasta}")

    for fasta in fasta_files:
        process_fasta(fasta)


if __name__ == "__main__":
    main()