import os
import glob
import re
import torch
import joblib
import numpy as np
import pandas as pd
from evo2 import Evo2

# ---------- config ----------
OUT_DIR = "."
os.makedirs(OUT_DIR, exist_ok=True)

LAYER_NAME = "blocks.28.mlp.l3"
DEVICE = "cuda"

WINDOW_SIZE = 5000
STRIDE = 500   

SCALER_PATH = "scaler.joblib"
PROBE_PATH = "best_linear_probe.pt"


# ---------- linear probe ----------
class LinearProbe(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


# ---------- load assets ----------
print("Loading scaler...")
scaler = joblib.load(SCALER_PATH)

print("Loading linear probe...")
probe = LinearProbe(4096)
probe.load_state_dict(torch.load(PROBE_PATH, map_location=DEVICE))
probe.to(DEVICE)
probe.eval()

print("Loading Evo2...")
model = Evo2("evo2_7b")
model.model.eval()


# ---------- fasta reader ----------
def read_fasta(path):
    headers = []
    sequences = []

    with open(path) as f:
        seq = []
        header = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    sequences.append("".join(seq))
                    headers.append(header)
                header = line[1:]
                seq = []
            else:
                seq.append(line)
        if header is not None:
            sequences.append("".join(seq))
            headers.append(header)

    return headers, sequences


# ---------- fixed windows ----------
def split_sequence_windows(seq, window_size=5000, stride=5000):
    seq = seq.replace(" ", "").upper()
    seq_len = len(seq)

    if seq_len == 0:
        return [], []

    windows = []
    starts = []

    if seq_len <= window_size:
        windows.append(seq)
        starts.append(0)
        return windows, starts

    for start in range(0, seq_len - window_size + 1, stride):
        end = start + window_size
        windows.append(seq[start:end])
        starts.append(start)

    # 保证最后一段覆盖到末尾
    last_start = seq_len - window_size
    last_window = seq[last_start:last_start + window_size]
    if len(windows) == 0 or windows[-1] != last_window:
        windows.append(last_window)
        starts.append(last_start)

    return windows, starts


# ---------- embedding ----------
def get_window_embedding(window_seq):
    window_seq = window_seq.replace(" ", "").upper()

    tokens = model.tokenizer.tokenize(window_seq)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        _, embeddings = model(
            input_ids,
            return_embeddings=True,
            layer_names=[LAYER_NAME]
        )

    layer_tensor = embeddings[LAYER_NAME]              # [1, L, D]
    vec = layer_tensor.mean(dim=1).squeeze(0)          # [D]

    out = vec.detach().float().cpu().numpy()

    del input_ids, embeddings, layer_tensor, vec
    torch.cuda.empty_cache()

    return out


# ---------- score one embedding ----------
def score_embedding(embedding_1d):
    emb2d = embedding_1d.reshape(1, -1)         # [1, D]
    emb2d = scaler.transform(emb2d)

    x = torch.tensor(emb2d, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        prob = torch.sigmoid(probe(x)).item()

    return float(prob)


# ---------- make output prefix ----------
def make_prefix_from_fasta(fasta_path):
    """
    例子：
    rice_Rloop_sequences.fa -> rice
    fly_Rloop_sequences.fa -> fly
    zebrafish.fa -> zebrafish
    """
    base = os.path.basename(fasta_path)
    stem = os.path.splitext(base)[0]

    stem = re.sub(r"_Rloop_sequences$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_sequences$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_Rloop$", "", stem, flags=re.IGNORECASE)

    return stem


# ---------- process one fasta ----------
def process_one_fasta(fasta_path):
    prefix = make_prefix_from_fasta(fasta_path)

    window_out_csv = os.path.join(OUT_DIR, f"{prefix}_raw.csv")
    gene_max_out_csv = os.path.join(OUT_DIR, f"{prefix}_summary.csv")

    headers, sequences = read_fasta(fasta_path)

    window_rows = []
    gene_rows = []

    print("=" * 80)
    print(f"Processing FASTA: {fasta_path}")
    print(f"Output prefix: {prefix}")
    print(f"Total genes: {len(sequences)}")

    for i, (header, seq) in enumerate(zip(headers, sequences), start=1):
        windows, starts = split_sequence_windows(
            seq,
            window_size=WINDOW_SIZE,
            stride=STRIDE
        )

        gene_scores = []

        for win_idx, (wseq, start) in enumerate(zip(windows, starts), start=1):
            emb = get_window_embedding(wseq)
            score = score_embedding(emb)

            end = start + len(wseq)
            gene_scores.append(score)

            window_rows.append({
                "source_fasta": os.path.basename(fasta_path),
                "species_prefix": prefix,
                "header": header,
                "gene_length": len(seq),
                "window_index": win_idx,
                "window_start": start,
                "window_end": end,
                "window_length": len(wseq),
                "score": score
            })

        max_score = max(gene_scores) if len(gene_scores) > 0 else np.nan
        mean_score = float(np.mean(gene_scores)) if len(gene_scores) > 0 else np.nan

        gene_rows.append({
            "source_fasta": os.path.basename(fasta_path),
            "species_prefix": prefix,
            "header": header,
            "gene_length": len(seq),
            "num_windows": len(gene_scores),
            "max_score": max_score,
            "mean_score": mean_score
        })

        print(
            f"Processed {i}/{len(sequences)} | "
            f"header={header} | seq_len={len(seq)} | windows={len(gene_scores)} | "
            f"max_score={max_score:.6f}"
        )

    df_window = pd.DataFrame(window_rows)
    df_gene = pd.DataFrame(gene_rows)

    df_window.to_csv(window_out_csv, index=False)
    df_gene.to_csv(gene_max_out_csv, index=False)

    print("Done.")
    print(f"Saved window-level scores to: {window_out_csv}")
    print(f"Saved gene-level max scores to: {gene_max_out_csv}")


# ---------- main ----------
def process_all_fastas():
    fasta_files = sorted(glob.glob("*.fa"))

    if len(fasta_files) == 0:
        print("No .fa files found in current directory.")
        return

    print(f"Found {len(fasta_files)} FASTA files:")
    for f in fasta_files:
        print(f"  - {f}")

    for fasta_path in fasta_files:
        process_one_fasta(fasta_path)


if __name__ == "__main__":
    process_all_fastas()