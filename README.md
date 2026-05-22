# RLoopBench

RLoopBench is a benchmark for evaluating whether DNA foundation models generalize to R-loop-forming sequence prediction. The benchmark compares rule-based features, classical sequence encodings, task-specific deep learning models, and DNA foundation model embeddings under a unified linear-probe evaluation framework.

R-loops are three-stranded nucleic acid structures consisting of an RNA--DNA hybrid and a displaced single-stranded DNA strand. Because R-loop formation is associated with transcription, replication stress, genome instability, and disease-related genome dysfunction, R-loop prediction provides a biologically distinct test case beyond conventional gene regulatory benchmarks.

## Repository overview

This repository provides:

- Example linear-probe training code using 3-mer encoding
- Pretrained linear-probe classifiers for each representation
- Inference scripts for applying trained linear probes

## Methods included

RLoopBench evaluates the following representation paradigms.

### Rule-based method

- QmRLFS-finder

Please refer to http://r-loop.org/?pg=qmrlfs for details.

### Classical sequence representations

- 3-mer frequency
- 4-mer frequency
- One-hot encoding

Trained models and inference scripts are provided in `models/`. These scripts can be run using the conda environment specified in `environment.yml`.

```bash
conda env create -f environment.yml
conda activate rloop_bench
```

We also provide an example workflow in `training/` showing how to generate 3-mer embeddings, visualize embeddings, and train a linear-probe classifier.

### Task-specific deep learning models

- DeepER: https://github.com/NjuChenlab/DeepER
- deepRloopPre: https://github.com/PEHGP/deepRloopPre

Please refer to the original repositories for installation and usage details.

### DNA foundation model embeddings

- Evo2: https://github.com/arcinstitute/evo2
- Nucleotide Transformer v3: https://huggingface.co/InstaDeepAI/NTv3_650M_pre
- DNABERT-2: https://github.com/MAGICS-LAB/DNABERT_2

The trained linear-probe models and reference inference scripts for these representations are provided in `models/`.

Please use the foundation-model inference scripts together with each DNA foundation model's own implementation environment. The scripts in this repository are intended as reference implementations for reproducing the embedding-to-linear-probe inference step.
