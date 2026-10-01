# Offline analyses for the reviewer comments (2026-10-01)

CPU only, no reader calls. Data: `results/hf_main_eval/` (labels/raw, eval). Scripts: `scripts/offline/`
(`a1_interactions.py`, `a2_a3_refit.py`, `a4_a7_eval_stats.py`, `common.py`). Outputs:
`results/offline_reviewer/*.json`. Primary reader Qwen3-8B unless noted. All of this is post hoc (after the
main run), so it is exploratory.

## A1: additivity (`a1_interactions.json`)
- Determinism: across ~12.5k duplicate-mask groups per multi-hop train source, F1 differs in ≤9 groups
  (log-prob never differs). The residual is not sampling noise.
- 5-fold CV R² (train docs): additive 0.536 / 0.549 (VIMQA / HotpotQA). Adding the gold-pair interaction:
  +0.033 / +0.034. All pairwise interactions (alpha chosen on dev): +0.031 / +0.036. Single-hop: all pairs
  +0.014 / +0.019, top-2 pair ≈ 0.
- Gold-pair interaction coefficient: median 0.025 (mean ≈ 0) vs. mean gold main effect 0.28–0.32.
- Takeaway: interactions exist on multi-hop but explain little of the ~45% residual.

## Sparse vs dense refit at equal mask count (`a2_a3_refit.json`, `a2`)
Answer chunk ranked first by beta: all masks 0.897 / 0.907 (VIMQA / HotpotQA); sparse only (≤2 kept, mean
18.6 masks) 0.783 / 0.816; same number of dense masks (5 draws) 0.800 / 0.812. The drop is driven by the
mask count. At equal count, sparse labels are no better. This replaces the paper's earlier 84%/82% figures.

## K curve, label quality only (`a2_a3_refit.json`, `kcurve`)
Answer-first rate (VIMQA/HotpotQA): K'=8 .65/.69, 16 .80/.81, 32 .86/.88, 48 .88/.90, all .90/.91.
Spearman with full beta: .46/.48, .59/.61, .77/.78, .89/.89. Pruners were not retrained.

## F1 vs log-prob labels (`a4_a7_eval_stats.json`, `a4`)
Paired cluster bootstrap, Holm over 10 cells: no cell significant. Largest is VIMQA 4x, +0.023
(p = 0.007 raw, 0.072 Holm). EM gives the same picture.

## Small-cluster robustness and MDE (`a5_a6`, `h3_mde`)
- H2b: cluster sign-flip and wild cluster bootstrap agree with the cluster bootstrap in all 4 cells
  (UIT-ViQuAD 4x still passes; sign-flip Holm p = 0.008). The TOST margin needed for 80% power is 0.015–0.031.
- H1-oracle (n = 100): MDE 0.036–0.098 F1 (one-sided, alpha 0.05, power 0.8, before Holm), versus the
  0.05 margin.
- H3: MDE of the difference in upgrade retention has median 0.22 (0.08–1.08) over 56 reader-pair cells.

## Budget-matched comparison (`a7`)
Interpolating ours_beta to EXIT's actual compression changes its F1 by only +0.004 to +0.012. At 8x the gap
to EXIT stays −0.067 / −0.036 / −0.100 (VIMQA / HotpotQA / 2Wiki). ours_sent and EXIT use 94–95% of the budget.
