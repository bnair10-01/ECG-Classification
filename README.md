# ECG Classification with Parameter-Efficient Fine-Tuning (PEFT)

Binary ECG segment classification (normal vs. abnormal) using pre-trained speech encoders (Wav2Vec2, HuBERT) with frozen backbones and two lightweight adaptation strategies: a trainable classifier head and LoRA. A third, small CNN backbone ("ECG-FM", see caveats) is run through the same pipeline as a non-speech baseline.

The point of the project is a **controlled comparison harness**: six configurations trained with identical data splits, loss, optimizer, early stopping and metrics. The headline result is modest (validation AUC ≈ 0.57–0.60, see [Results](#results)), and this README states plainly what the numbers do and do not show.

## Task and dataset

| | |
|---|---|
| File | `df_segment2.csv` (provided with the course lab; the original public source is not documented here) |
| Size | 65,424 segments x 600 samples, single lead, no header row; last column is the label |
| Label | Binary, normal vs. abnormal. The 0/1 encoding is not documented by the provider; metrics treat `1` as the positive class |
| Class balance | Close to 50/50 (validation split: 4,923 vs. 4,891) |
| Split | Stratified 70 / 15 / 15 (train / val / test), `random_state=42` |

The data file is not included in this repository.

## Configurations

| # | Backbone | Adaptation | What is actually trained |
|---|----------|-----------|--------------------------|
| 1 | `facebook/wav2vec2-base` (frozen) | **Head** | MLP head `Linear(768,256) → ReLU → Dropout(0.1) → Linear(256,1)` on mask-aware mean-pooled features (~197K params, ~0.2%) |
| 2 | `facebook/wav2vec2-base` | **LoRA** (`peft`) | r=8, alpha=16, dropout 0.1 on `q_proj, k_proj, v_proj, out_proj` (~0.59M params, ~0.6%) |
| 3 | `facebook/hubert-base-ls960` (frozen) | **Head** | Same MLP head as #1 |
| 4 | `facebook/hubert-base-ls960` | **LoRA** (`peft`) | Same LoRA config as #2 |
| 5 | 3-layer 1-D CNN (frozen) | **Head** | MLP head on 128-d features (~33K of ~206K params, ~16%) |
| 6 | 3-layer 1-D CNN | **LoRA** (custom) | See caveat below |

Terminology: the "Adapter" runs are a **trainable classifier head on a frozen encoder** (a linear-probe-style MLP). They are *not* bottleneck adapter modules inserted inside the transformer layers (Houlsby-style). The LoRA runs use the Hugging Face `peft` library for Wav2Vec2/HuBERT and a small hand-written LoRA layer (`LRL` / `apply_lora`) for the CNN.

## Pipeline

1. **Normalize**: row-wise z-score per segment (before splitting).
2. **Resample**: linear interpolation from 600 to 1000 samples.
3. **Collate**: per-batch z-score, then the Hugging Face feature extractor (`sampling_rate=16000`, `do_normalize=True`). The 16 kHz value is a label for the processor; the signal is not upsampled to 16 kHz, so the encoders see 1000 samples (roughly 2 encoder frames for Wav2Vec2/HuBERT).
4. **Train**: AdamW, BCE-with-logits, batch size 64, max 10 epochs, early stopping on validation loss (patience 3) with the best checkpoint restored.
   - Heads: lr 5e-5. LoRA: lr 2.5e-5. Weight decay 1e-4.
5. **Evaluate**: accuracy, precision, recall, F1, AUC-ROC on validation (every epoch) and validation + test (final). Decision threshold 0.5.
6. **Report**: CSV summaries, per-model training curves, cross-model comparison curves, validation confusion matrices.

Other details: seeds fixed (`SEED = 42` for Python, NumPy, PyTorch, CUDA; GPU kernels are not forced deterministic), device auto-selection CUDA → MPS → CPU.

## Results

Validation set, n = 9,814. Single run per configuration (one seed).

| Model | Acc | Precision | Recall | F1 | AUC (≈) |
|-------|----:|----------:|-------:|---:|--------:|
| Wav2Vec2 + Head | 0.554 | 0.545 | 0.636 | 0.587 | 0.58 |
| Wav2Vec2 + LoRA | 0.566 | 0.580 | 0.469 | 0.519 | 0.60 |
| HuBERT + Head | 0.537 | 0.526 | 0.711 | 0.605 | 0.57 |
| HuBERT + LoRA | 0.555 | 0.546 | 0.633 | 0.587 | 0.59 |
| ECG-FM (CNN) + Head | 0.556 | 0.567 | 0.465 | 0.511 | 0.57 |
| ECG-FM (CNN) + LoRA | 0.536 | 0.595 | 0.213 | 0.314 | 0.57 |

Accuracy/precision/recall/F1 are computed from the saved validation confusion matrices (best-val-loss checkpoint). AUC values are read off the validation-AUC curves at the final epoch, so treat them as approximate (±0.005). Early stopping ended Wav2Vec2 + Head after epoch 6 and HuBERT + LoRA after epoch 5. Your own run writes exact numbers to `results.csv` and `detailed.csv`.

**What the results support**
- All six configurations beat chance only slightly (balanced classes, chance AUC = 0.5). With 600-sample single-lead input and frozen speech encoders, there is weak but real signal.
- Wav2Vec2 + LoRA has the best AUC (≈0.60) and the lowest, steadily decreasing validation loss. LoRA variants on the two speech encoders (≈0.59–0.60) edge out the head-only variants (≈0.57–0.58).
- Head-only models trade precision for recall (HuBERT + Head: recall 0.71, precision 0.53), i.e. they lean toward predicting "positive".

**What they do not support**
- The AUC gaps between methods are about 0.01–0.03 from a single seed with no confidence intervals. "LoRA beats heads" is suggestive, not established.
- The ECG-FM rows are not a meaningful foundation-model comparison (see below): their validation loss stays ≈0.69 and AUC stays flat ≈0.567 across epochs.

## Known limitations

- **ECG-FM is a stand-in.** The "ECG-FM" backbone is a small 3-layer CNN. The script tries to load `data/mimic_iv_ecg_physionet_pretrained.pt` with `strict=False`, which will not populate this architecture from a real ECG foundation-model checkpoint, so the backbone is effectively randomly initialised. Read these rows as a frozen-random-CNN baseline, not as a pre-trained ECG model.
- **ECG-FM + LoRA injects no LoRA.** `apply_lora` only wraps `nn.Linear` layers and the CNN has none, so this configuration is the head-only model with a different learning rate.
- **LoRA classifier head (to verify).** The Wav2Vec2/HuBERT LoRA models use `get_peft_model` without `task_type` or `modules_to_save`, so the randomly initialised sequence-classification head is probably frozen and only the LoRA matrices train. Check the `print_trainable_parameters()` output; setting `modules_to_save=["projector", "classifier"]` is the likely fix.
- **Very short input to the encoders.** 1000 samples yields only ~2 frames in Wav2Vec2/HuBERT, which likely caps what these models can extract.
- **Triple normalization** (row-wise, per-batch, processor) is redundant.
- **Tuning mode is limited.** `TUNING_MODE = True` grid-searches batch size x learning rate x weight decay (8 runs) on Wav2Vec2 + Head only, scores by mean of final-epoch val accuracy and AUC, and writes `metrics/hyperparameter_tuning.csv`. The best setting is not fed back into the main run. `lora_ranks`, `adapter_hiddens` and `learning_rates_lora` in `HP_CONFIG` are unused.
- Single seed; no cross-validation; no patient-level split information (if segments from one patient appear in several splits, scores could be optimistic).

## Running

The script was written for Google Colab and uses `base = "/content"` for all inputs and outputs. To run elsewhere, change `base` near the top of `bnair_lab3-1.py` to a writable directory.

```bash
pip install torch transformers peft scikit-learn pandas numpy matplotlib
# place df_segment2.csv (or df_segment2.csv.zip) in <base>/
python bnair_lab3-1.py
```

Package versions used for the reported runs were not recorded. A GPU (CUDA or Apple MPS) is strongly recommended; the encoders are downloaded from the Hugging Face Hub on first run.

Optional: put `mimic_iv_ecg_physionet_pretrained.pt` in `<base>/data/` (see ECG-FM caveat).

### Outputs (written under `base`)

```
results.csv                  # Val acc / F1 / AUC and test acc per model
detailed.csv                 # Acc, precision, recall, F1, AUC for val and test splits
histories/<model>.csv        # epoch, train_loss, val_loss, val_acc, val_precision,
                             #   val_recall, val_f1, val_auc
plots/<model>_curves.png     # 2x2 training curves per model
plots/comparison_<metric>.png  # all six models overlaid (val_acc, val_loss, val_f1, val_auc)
plots/<model>_Val.png        # validation confusion matrices (counts + row %)
metrics/hyperparameter_tuning.csv   # only when TUNING_MODE = True
```

## Possible next steps

- Fix the LoRA head training and the ECG-FM LoRA no-op, then re-run with multiple seeds.
- Use a real pre-trained ECG encoder, and feed longer or multi-lead inputs.
- Report test-set metrics with confidence intervals; add a patient-level split if IDs are available.
- Add true bottleneck adapters inside the transformer blocks for a like-for-like comparison with LoRA.

## References

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022. https://arxiv.org/abs/2106.09685
- Baevski et al., *wav2vec 2.0*, NeurIPS 2020. https://arxiv.org/abs/2006.11477
- Hsu et al., *HuBERT*, 2021. https://arxiv.org/abs/2106.07447
- Houlsby et al., *Parameter-Efficient Transfer Learning for NLP*, ICML 2019 (background for adapter modules; not implemented here). https://arxiv.org/abs/1902.00751

## License

MIT, see [LICENSE](LICENSE).
