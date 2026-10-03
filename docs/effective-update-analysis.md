# Effective LoRA update analysis

This optional CPU analysis measures native FLUX.1 LoRA parameter updates. It does not score images, establish semantic compatibility, infer a person's identity or automatically assign block weights. The ordinary catalogue and Studio remain usable without Torch. App integration and explained experimental proposals are subsequent work.

For a native module with factors B and A, the effective update is `scale × B × A`. Scale is `alpha / rank` when alpha is present, otherwise one. Small Gram matrices provide squared Frobenius norms and signed pairwise inner products without constructing a full dense update. Norms are accumulated per canonical BASE/double/single slot; cosine is null for a zero-norm comparison. Opposite signed updates retain negative alignment.

The adapter supports the native two-dimensional factor pairs accepted by the conditional `flux1-dev-native-v1` target contract. Unknown variants fail explicitly. Local Comfy/loader source pins and bounded file-header/stat identities are checked before and after reading. This detects observed changes; it is not a full-file cryptographic content hash or proof of a loaded checkpoint.

## Runtime and limits

The separate project analysis environment uses the pinned `requirements-analysis.txt`: CPU Torch 2.9.1, safetensors 0.7.0 and NumPy 2.3.5. No Comfy environment is changed. Accumulation is CPU float64 with two threads by default. Defaults limit a request to eight LoRAs, rank 256, 4,096 modules, 1 GiB of logical tensor reads, five billion multiply-adds and 180 seconds checked between bounded operations. A limit, changed source, invalid tensor or cancellation produces no partial successful metrics.

Run on **Bender / PowerShell**, using real existing input paths:

```powershell
Set-Location 'E:\LoRA Project'
.\.venv-analysis\Scripts\python.exe -B tools\analyse_effective_lora.py 'E:\models\loras\first.safetensors' 'E:\models\loras\second.safetensors' --output '.local\effective-analysis\comparison.json'
```

The two input paths above are placeholders. The command reads inputs only and writes an atomic JSON receipt to the chosen output. Successful receipts include runtime, target/source pins, ordered identities, slot norms, pair measurements and work accounting. Failed receipts are explicitly marked with a reason code and null metrics. Receipts belong in local state, not public source control.

## Verification on 3 October 2026

- Full backend suite under the pinned CPU runtime: **237 passed**, including **26 focused metric tests**, with no optional tensor skips.
- Independent dense matrix oracles cover mixed rank, signed/absent alpha, chunking, exact negative alignment, zero norms, failure and resource boundaries. The oracle computes scaling independently of production code.
- Independent review of the Gram calculation found no remaining blockers.
- A read-only real person/clothing/style selection measured 304 native target modules in each of three LoRAs, three pairs, and 210,591,744 multiply-adds in approximately 0.85 seconds on this capture. This is a sample timing, not a performance guarantee. Private receipt: `.local/effective-analysis/person-clothing-style.json`.
- A larger sample correctly exceeded its work budget and returned no partial metrics.

The local launcher separately passed seven offline tests and a copied-database build, start, UI/API access, status, stop, restart and stop rehearsal. Original catalogue database SHA-256 remained `9af0501b6aab2d9aabb63e30bb36062b0a76c0eb61bb84f2a136dcf0330a2d35`. See [local-launcher.md](local-launcher.md).

Hosted backend/UI CI passed in [run 37154332301](https://github.com/clink2012/LoRA_weights_builder/actions/runs/37154332301). The separately dispatched [real CPU run 37154331576](https://github.com/clink2012/LoRA_weights_builder/actions/runs/37154331576) also passed its actual Torch/safetensors/NumPy preflight and complete backend suite. Both ran against `ced1bf319b38013103461684295e5d5336924a3a`; PR #79 merged as `e0d191f`.
