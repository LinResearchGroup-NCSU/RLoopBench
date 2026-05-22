# RLoopBench

RLoopBench is a benchmark for evaluating whether DNA sequence representations generalize to R-loop-forming sequence prediction. The benchmark compares rule-based features, classical sequence encodings, task-specific deep learning models, and DNA foundation model embeddings under a unified linear-probe evaluation framework.

R-loops are three-stranded nucleic acid structures consisting of an RNA--DNA hybrid and a displaced single-stranded DNA strand. Because R-loop formation is associated with transcription, replication stress, genome instability, and disease-related genome dysfunction, R-loop prediction provides a biologically distinct test case beyond conventional gene regulatory benchmarks.

## Repository overview

This repository provides:

- Processed R-loop benchmark datasets
- Feature generation scripts for k-mer and one-hot representations
- Example training code using one-hot encoding
- Pretrained linear-probe classifiers for each representation
- Inference scripts for applying trained linear probes
- Links and configuration details for DNA foundation model embeddings
- Data tables and scripts used to reproduce the main figures

## Methods included

RLoopBench evaluates the following representation paradigms:

### Rule-based method

- QmRLFS-finder

### Classical sequence representations

- 3-mer frequency
- 4-mer frequency
- One-hot encoding

### Task-specific deep learning models

- DeepER
- deepRloopPre

### DNA foundation model embeddings

- Evo2
- Nucleotide Transformer v3, NTv3
- DNABERT-2

All learned sequence representations are evaluated using a unified linear-probe classifier to reduce the influence of downstream model complexity.

## Repository structure

```text
RLoopBench/
├── data/                  # Raw and processed R-loop datasets
├── models/                # Trained linear-probe classifiers and scalers
├── scripts/               # Preprocessing, training, inference, and evaluation scripts
├── results/               # Benchmark metrics, predictions, and figure source data
├── figures/               # Plotting scripts for manuscript figures
├── notebooks/             # Example notebooks
└── docs/                  # Additional dataset and model documentation
