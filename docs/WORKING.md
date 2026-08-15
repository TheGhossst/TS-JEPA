# Working overlay (not paper-faithful)

The published TS-JEPA recipe (1 ms stride, cosine-only loss, Table II SGD 0.2)
makes the predictor copy \(z_k\) and the actor collapse toward the mean command.
This overlay is a **salvage recipe**. It does **not** claim IEEE headline numbers
(74.48% control accuracy, NMAE 0.004).

Paper-faithful code stays at [`configs/ts_jepa_baseline.yaml`](../configs/ts_jepa_baseline.yaml)
and [`docs/plan.md`](plan.md). Do not point `assert_plan_*` at this overlay.

## How to run

```powershell
python scripts/pipeline/run_working.py --config configs/ts_jepa_working.yaml --device cuda
```

Seed 0 only by default. Full 5-seed protocol: `--all-seeds`.

Data: `data_working/`. Checkpoints: `runs/ts_jepa_working/`, `runs/semantic_actor_working/`.

## Deviations from the paper

| Paper | Working overlay |
|-------|-----------------|
| \(\tau_o=1\) ms stored samples | Physics `dt=0.001`, **store / decide every 20 ms** (`observation_stride_steps=20`, `dp_substeps=20`) |
| \(\Psi(x_k)\) one RGB frame | **κ=2 channel-concat** into Ψ (`jepa_observation: kappa_stack`, 6 channels) so \(z\) can carry velocity |
| Cosine loss only | Cosine **+ VICReg** on raw context embeddings (\(\mu=25\), \(\nu=5\), \(\gamma=1\)). Train applies VICReg on the **effective batch** (256), not each microbatch of 16. Logged `vicreg_var` is \(\mathrm{mean}(\mathrm{relu}(\gamma-\mathrm{std}))\), not the std. |
| Encoder L2-normalize (IC) | **Off** — cosine still normalizes inside the loss; VICReg \(\gamma=1\) needs unbounded \(z\) |
| SGD LR 0.2 | **AdamW peak LR \(3\times10^{-4}\)** (5-epoch warmup) |
| 100 steps = 100 ms | 100 stored steps = **2 s** |

`plan.enforce: false` makes every `assert_plan_*` a no-op for this config.

## Success gates

Not paper metrics. Fail the working pipeline if any fail:

1. Predictor uses \(u\): mean cosine \(P(z,+20)\) vs \(P(z,-20)\) **< 0.95** and mean L2 **> 0.2**
2. Encoder effective rank **> 8**
3. Actor test NMAE **beats** the train-mean command baseline
4. Closed-loop Eq. 28 score over 100 stored steps, with a packet every \(K_p\) steps, **beats hold-last and zero-action** on the same inits

If gate 1 still fails after a short JEPA run, increase `observation_stride_steps` to **50** before a 150-epoch train. 5 ms is still subpixel and is not a fix.

## What this is not

- Not a reproduction of Girgis, Valcarce, Bennis, *IEEE IoT J.* 2026
- Not wireless Figs. 9–11
- Not GE-JEPA / burst channels
