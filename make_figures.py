"""
Regenerate the report figures from data we ALREADY have no model, no MEG data,
no re-run. Needs only matplotlib (+ numpy). Run:  python make_figures.py
Produces, in the current directory:
  - faithful_overfit.png   (Fig. 1 in the report: train -> 1.0, val flat at chance)
  - model_comparison.png   (Table 2 as grouped bars w/ error bars)
  - bandpass_ablation.png  (Table 1: band-pass collapses cross-subject)
"""
import re
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 1) Faithful AA-CascadeNet overfit curve — verbatim 100-epoch diagnostic log
#    (lr=1e-4, 100 epochs). Parsed from the pasted training output.
# ----------------------------------------------------------------------------
LOG = """
epoch   1 | train 1.389/0.255 | val 1.388/0.250
epoch   2 | train 1.389/0.233 | val 1.388/0.250
epoch   3 | train 1.387/0.272 | val 1.387/0.250
epoch   4 | train 1.388/0.248 | val 1.387/0.243
epoch   5 | train 1.385/0.283 | val 1.387/0.235
epoch   6 | train 1.383/0.288 | val 1.386/0.279
epoch   7 | train 1.380/0.319 | val 1.386/0.265
epoch   8 | train 1.373/0.355 | val 1.386/0.287
epoch   9 | train 1.361/0.384 | val 1.385/0.265
epoch  10 | train 1.328/0.429 | val 1.390/0.287
epoch  11 | train 1.292/0.430 | val 1.407/0.265
epoch  12 | train 1.247/0.440 | val 1.406/0.287
epoch  13 | train 1.170/0.475 | val 1.469/0.272
epoch  14 | train 1.092/0.478 | val 1.567/0.243
epoch  15 | train 1.038/0.509 | val 1.658/0.257
epoch  16 | train 0.981/0.515 | val 1.754/0.250
epoch  17 | train 0.904/0.575 | val 1.801/0.309
epoch  18 | train 0.870/0.562 | val 1.853/0.309
epoch  19 | train 0.851/0.571 | val 1.948/0.287
epoch  20 | train 0.792/0.619 | val 2.070/0.265
epoch  21 | train 0.752/0.616 | val 2.112/0.272
epoch  22 | train 0.740/0.648 | val 2.130/0.279
epoch  23 | train 0.723/0.643 | val 2.189/0.265
epoch  24 | train 0.687/0.669 | val 2.383/0.279
epoch  25 | train 0.651/0.701 | val 2.281/0.301
epoch  26 | train 0.614/0.702 | val 2.410/0.265
epoch  27 | train 0.561/0.730 | val 2.543/0.279
epoch  28 | train 0.578/0.712 | val 2.693/0.250
epoch  29 | train 0.542/0.727 | val 2.810/0.250
epoch  30 | train 0.524/0.743 | val 2.776/0.265
epoch  31 | train 0.491/0.765 | val 2.993/0.235
epoch  32 | train 0.486/0.777 | val 2.799/0.279
epoch  33 | train 0.467/0.776 | val 2.871/0.272
epoch  34 | train 0.456/0.789 | val 2.911/0.294
epoch  35 | train 0.404/0.842 | val 2.912/0.294
epoch  36 | train 0.393/0.842 | val 3.104/0.272
epoch  37 | train 0.358/0.848 | val 3.130/0.287
epoch  38 | train 0.371/0.853 | val 3.128/0.257
epoch  39 | train 0.361/0.854 | val 3.331/0.272
epoch  40 | train 0.354/0.853 | val 3.250/0.287
epoch  41 | train 0.302/0.879 | val 3.226/0.279
epoch  42 | train 0.278/0.906 | val 3.310/0.265
epoch  43 | train 0.273/0.913 | val 3.480/0.250
epoch  44 | train 0.238/0.929 | val 3.531/0.250
epoch  45 | train 0.225/0.923 | val 3.573/0.257
epoch  46 | train 0.202/0.935 | val 3.595/0.287
epoch  47 | train 0.175/0.958 | val 3.755/0.257
epoch  48 | train 0.147/0.960 | val 3.735/0.272
epoch  49 | train 0.139/0.962 | val 3.814/0.272
epoch  50 | train 0.116/0.972 | val 4.088/0.221
epoch  51 | train 0.119/0.967 | val 4.009/0.250
epoch  52 | train 0.112/0.974 | val 4.153/0.228
epoch  53 | train 0.098/0.975 | val 4.118/0.221
epoch  54 | train 0.067/0.988 | val 4.216/0.279
epoch  55 | train 0.078/0.983 | val 4.475/0.257
epoch  56 | train 0.061/0.994 | val 4.468/0.279
epoch  57 | train 0.073/0.980 | val 4.539/0.272
epoch  58 | train 0.070/0.987 | val 4.552/0.250
epoch  59 | train 0.067/0.984 | val 4.588/0.243
epoch  60 | train 0.062/0.988 | val 4.546/0.279
epoch  61 | train 0.060/0.988 | val 4.627/0.287
epoch  62 | train 0.056/0.990 | val 4.845/0.257
epoch  63 | train 0.050/0.990 | val 5.009/0.257
epoch  64 | train 0.047/0.989 | val 5.141/0.250
epoch  65 | train 0.057/0.982 | val 4.941/0.243
epoch  66 | train 0.052/0.985 | val 5.070/0.265
epoch  67 | train 0.033/0.998 | val 5.162/0.243
epoch  68 | train 0.039/0.990 | val 5.134/0.257
epoch  69 | train 0.035/0.990 | val 4.999/0.265
epoch  70 | train 0.029/0.991 | val 5.028/0.235
epoch  71 | train 0.030/0.995 | val 5.210/0.250
epoch  72 | train 0.018/0.999 | val 5.246/0.243
epoch  73 | train 0.038/0.994 | val 5.548/0.228
epoch  74 | train 0.027/0.996 | val 5.254/0.265
epoch  75 | train 0.021/0.998 | val 5.185/0.287
epoch  76 | train 0.030/0.991 | val 5.475/0.265
epoch  77 | train 0.026/0.993 | val 5.511/0.250
epoch  78 | train 0.023/0.994 | val 5.579/0.265
epoch  79 | train 0.023/0.994 | val 5.686/0.257
epoch  80 | train 0.049/0.985 | val 5.268/0.243
epoch  81 | train 0.046/0.989 | val 5.175/0.287
epoch  82 | train 0.053/0.985 | val 5.287/0.279
epoch  83 | train 0.048/0.988 | val 5.259/0.257
epoch  84 | train 0.035/0.989 | val 5.157/0.257
epoch  85 | train 0.041/0.993 | val 5.129/0.309
epoch  86 | train 0.022/0.993 | val 5.217/0.294
epoch  87 | train 0.034/0.994 | val 5.205/0.316
epoch  88 | train 0.031/0.993 | val 5.386/0.294
epoch  89 | train 0.032/0.989 | val 5.352/0.279
epoch  90 | train 0.028/0.991 | val 5.517/0.279
epoch  91 | train 0.046/0.980 | val 5.797/0.287
epoch  92 | train 0.029/0.990 | val 5.786/0.243
epoch  93 | train 0.033/0.993 | val 5.688/0.294
epoch  94 | train 0.023/0.993 | val 5.896/0.243
epoch  95 | train 0.030/0.988 | val 5.465/0.235
epoch  96 | train 0.017/0.995 | val 5.759/0.287
epoch  97 | train 0.020/0.993 | val 5.381/0.272
epoch  98 | train 0.012/1.000 | val 5.764/0.243
epoch  99 | train 0.033/0.991 | val 5.347/0.287
epoch 100 | train 0.035/0.994 | val 5.957/0.265
"""

rows = re.findall(r"train ([\d.]+)/([\d.]+) \| val ([\d.]+)/([\d.]+)", LOG)
tr_loss, tr_acc, va_loss, va_acc = (np.array(c, float) for c in zip(*rows))
ep = np.arange(1, len(tr_acc) + 1)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(ep, tr_acc, label="train", color="#1f77b4")
ax[0].plot(ep, va_acc, label="validation", color="#ff7f0e")
ax[0].axhline(0.25, ls="--", lw=.8, c="gray", label="chance")
ax[0].set_ylim(0, 1.03); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("accuracy")
ax[0].set_title("Accuracy"); ax[0].legend(loc="center right", fontsize=8); ax[0].grid(alpha=.3)

ax[1].plot(ep, tr_loss, label="train", color="#1f77b4")
ax[1].plot(ep, va_loss, label="validation", color="#ff7f0e")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("loss"); ax[1].set_title("Loss")
ax[1].legend(loc="upper left", fontsize=8); ax[1].grid(alpha=.3)

fig.suptitle("Faithful AA-CascadeNet (~2.1M params): train accuracy $\\to$ 1.0, "
             "validation stuck at chance", fontsize=10)
fig.tight_layout()
fig.savefig("faithful_overfit.png", dpi=200, bbox_inches="tight")
print("wrote faithful_overfit.png")

# ----------------------------------------------------------------------------
# 2) Main comparison (Table 2): file-level accuracy, mean +/- std, no band-pass.
# ----------------------------------------------------------------------------
models = ["AA-CascadeNet\n(compact 47k)", "CascadeFormer\n(compact 138k)",
          "EEG Conformer\n(87k)", "AA-CascadeNet\n(faithful 2.1M)",
          "CascadeFormer\n(faithful 2.1M)"]
intra_m = [0.775, 0.850, 0.750, 0.550, 0.550]
intra_s = [0.094, 0.122, 0.137, 0.127, 0.100]
cross_m = [0.667, 0.696, 0.667, 0.496, 0.550]
cross_s = [0.070, 0.010, 0.048, 0.036, 0.047]

x = np.arange(len(models)); w = 0.38
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(x - w/2, intra_m, w, yerr=intra_s, capsize=3, label="intra-subject", color="#4C72B0")
ax.bar(x + w/2, cross_m, w, yerr=cross_s, capsize=3, label="cross-subject", color="#C44E52")
ax.axhline(0.25, ls="--", lw=.8, c="gray", label="chance")
ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8)
ax.set_ylabel("file-level accuracy (mean $\\pm$ std, 5 seeds)")
ax.set_ylim(0, 1.0); ax.set_title("Model comparison (no band-pass)")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig("model_comparison.png", dpi=200, bbox_inches="tight")
print("wrote model_comparison.png")

# ----------------------------------------------------------------------------
# 3) Band-pass ablation (Table 1): cross-subject collapses with band-pass.
# ----------------------------------------------------------------------------
bp_models = ["AA-CascadeNet\n(compact)", "CascadeFormer\n(compact)", "EEG Conformer"]
cross_bp   = [0.296, 0.271, 0.392]   # with [1,45] Hz band-pass
cross_nobp = [0.667, 0.696, 0.667]   # minimal pipeline

x = np.arange(len(bp_models)); w = 0.38
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x - w/2, cross_bp,   w, label="with [1,45] Hz band-pass", color="#999999")
ax.bar(x + w/2, cross_nobp, w, label="no band-pass (minimal)",  color="#55A868")
ax.axhline(0.25, ls="--", lw=.8, c="gray", label="chance")
ax.set_xticks(x); ax.set_xticklabels(bp_models, fontsize=8)
ax.set_ylabel("cross-subject file-level accuracy")
ax.set_ylim(0, 0.8); ax.set_title("Band-pass collapses cross-subject transfer")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig("bandpass_ablation.png", dpi=200, bbox_inches="tight")
print("wrote bandpass_ablation.png")

print("done — 3 figures written to the current directory.")
