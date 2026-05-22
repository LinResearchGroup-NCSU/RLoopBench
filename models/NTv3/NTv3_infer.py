import math
import os
import glob

import torch
import numpy as np
import pandas as pd
import joblib

from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM


MODEL_PATH = "InstaDeepAI/NTv3_650M_pre_8kb"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WINDOW_SIZE = 5120
STEP_SIZE = 512

SCALER_PATH = "scaler.joblib"
PROBE_PATH = "best_linear_probe.pt"
PROBE_INPUT_DIM = 1536


class LinearProbe(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


print("Loading tokenizer and NTv3 model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
nt_model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
nt_model.eval()
nt_model.to(DEVICE)

print("Loading scaler and linear probe...")
scaler = joblib.load(SCALER_PATH)

probe = LinearProbe(PROBE_INPUT_DIM)
probe.load_state_dict(torch.load(PROBE_PATH, map_location="cpu"))
probe.eval()
probe.to(DEVICE)


def masked_mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


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


def get_ntv3_embedding(chunk):
    chunk_len = len(chunk)
    padded_len = math.ceil(chunk_len / 128) * 128

    inputs = tokenizer(
        chunk,
        return_tensors="pt",
        add_special_tokens=False,
        padding="max_length",
        max_length=padded_len,
        return_attention_mask=True,
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = nt_model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        pooled = masked_mean_pool(last_hidden_state, inputs["attention_mask"])

    return pooled.squeeze(0).cpu().numpy()


def score_embedding(embedding_1d):
    x = embedding_1d.reshape(1, -1)
    x_scaled = scaler.transform(x)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        prob = torch.sigmoid(probe(x_tensor)).item()

    return float(prob)


def process_fasta(fasta_file):
    print(f"\nProcessing {fasta_file} ...")

    window_rows = []
    seq_rows = []

    total_records = 0
    total_windows = 0

    for record in tqdm(SeqIO.parse(fasta_file, "fasta"), desc=f"Scanning {fasta_file}"):
        header = record.id
        full_sequence = str(record.seq).upper()
        seq_len = len(full_sequence)

        if seq_len == 0:
            continue

        total_records += 1

        windows, starts = split_sequence_windows(
            full_sequence,
            window_size=WINDOW_SIZE,
            step_size=STEP_SIZE
        )

        seq_scores = []

        for chunk, start in zip(windows, starts):
            end = start + len(chunk)

            emb = get_ntv3_embedding(chunk)
            score = score_embedding(emb)

            window_rows.append({
                "header": header,
                "sequence_length": seq_len,
                "window_start_0based": start,
                "window_end_0based_exclusive": end,
                "window_length": len(chunk),
                "score": score
            })

            seq_scores.append(score)
            total_windows += 1

        seq_rows.append({
            "header": header,
            "sequence_length": seq_len,
            "num_windows": len(seq_scores),
            "max_score": float(np.max(seq_scores)),
            "mean_score": float(np.mean(seq_scores))
        })

    base = os.path.splitext(os.path.basename(fasta_file))[0]
    window_csv = f"{base}_raw.csv"
    summary_csv = f"{base}_summary.csv"

    pd.DataFrame(window_rows).to_csv(window_csv, index=False)
    pd.DataFrame(seq_rows).to_csv(summary_csv, index=False)

    print(f"Done: {fasta_file}")
    print(f"Total sequences: {total_records}")
    print(f"Total windows: {total_windows}")
    print(f"Saved window scores to: {window_csv}")
    print(f"Saved summary to: {summary_csv}")


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