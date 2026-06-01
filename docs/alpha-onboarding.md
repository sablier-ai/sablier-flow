# Alpha onboarding — sablier-flow

If you're reading this, you have early access to the sablier-flow alpha.
This page is the 5-minute onboarding to actually use the SDK against
the live alpha endpoint.

## What you need

- Python 3.10 / 3.11 / 3.12
- An API key (Sablier sends you one)
- 5 minutes

## 1. Install the SDK

```bash
pip install 'sablier-flow[adapters-backtrader,adapters-vectorbt]'
```

## 2. Get the live endpoint config

```bash
# One command fetches the endpoint URL + pinned cert from GCS and
# prints copy-pasteable env-var exports. Requires gcloud in PATH.
sablier-flow setup
# … then paste the printed `export ...` block into your shell, OR:
eval "$(sablier-flow setup | grep '^export ')"
```

The cert is self-signed RSA-4096 valid 90 days, subject = the VM's
current external IP. Your SDK pins it via the `SABLIER_FLOW_CERT` env
var (`Client(verify=...)` under the hood).

## 3. The five lines

```python
import pandas as pd
import sablier_flow

# 3a. Load YOUR data (pandas DataFrame, DatetimeIndex, numeric columns)
real = pd.read_parquet("my_universe.parquet")

# 3b. Connect to the alpha endpoint
client = sablier_flow.Client(
    api_key="sk-...",                       # your alpha key from Sablier
    endpoint=open("endpoint.url").read().strip(),
    pinned_image_digest="sha256:" + "0" * 64,  # alpha uses the dev digest
    attestation_mode="fake-for-dev",        # alpha is L4 (non-confidential)
    verify="./alpha-server.crt",            # pin the self-signed cert
)

# 3c. Generate N synthetic alternative versions of your trading window.
# Runs a real Sablier flow model on a real L4 GPU. ~2-3 min train + gen.
synthetic = client.alternative_versions(real, n_paths=200, horizon=252)

# 3d. Run YOUR existing backtest unchanged on each synthetic version
from sablier_flow.adapters import as_dataframes
synth_dfs = as_dataframes(synthetic, index=real.index[-252:])
real_pnl   = my_backtest(real.tail(252))
synth_pnls = [my_backtest(df) for df in synth_dfs]

# 3e. The smoking gun
report = sablier_flow.robustness(real_pnl, synth_pnls)
print(report.verdict, report.overfit_score)
```

## 4. What you get back

`synthetic` is a `GenerationResult` with these fields:

| Field | What it tells you |
|---|---|
| `paths_returns`, `paths_prices` | The synthetic data (numpy arrays) |
| `feature_names`, `horizon`, `n_paths` | Shape metadata |
| **`memorization_risk`** | `'low'` / `'medium'` / `'high'` — is the model just regurgitating your training data? |
| **`memorization_nn_distance_ratio`** | Synth-to-train / train-to-train NN distance. `> 0.80` = low risk; `0.50–0.80` = medium; `< 0.50` = memorized. |
| **`validation_overall`** | `'pass'` / `'warn'` / `'fail'` — is the synthetic distribution structurally consistent with your input? |
| **`validation_metrics`** | Per-metric breakdown: marginal moments, cross-asset correlation, lag-1 autocorr |

`report` is a `RobustnessReport` with:

| Field | What it tells you |
|---|---|
| **`verdict`** | `'robust'` / `'borderline'` / `'overfit'` / `'highly_overfit'` |
| **`overfit_score`** | 0.0–1.0. Higher = more likely overfit. |
| `synthetic_median`, `synthetic_p5`, `synthetic_p95` | Distribution stats of your backtest across the synthetic alternatives |
| `notes` | Human-readable annotations on the verdict |

## 5. What's actually happening

Every alpha job runs through the full production wire protocol:

```
your laptop                              alpha endpoint (g2 + NVIDIA L4)
   │                                            │
   │  1. POST /v1/jobs                          │  → JobStore allocates job_id +
   │                                            │     fresh per-job ephemeral X25519 keypair
   │  ◄── 2. attestation quote                  │  ← quote bound to ephemeral pubkey
   │                                            │
   │  3. envelope-encrypt YOUR data             │
   │      with the ephemeral pubkey             │
   │  ──> PUT /v1/jobs/{id}/data ──────────────►│  → TEE decrypts inside the enclave,
   │                                            │     parses Parquet, trains FLOW model on
   │  ◄── 4. polling, status='running'/'completed'│   GPU (~2-3 min), generates N paths,
   │                                            │     runs memorization + validation checks,
   │                                            │     AES-GCM-encrypts the result
   │                                            │
   │  ──> GET /v1/jobs/{id}/result ────────────►│  ← returns the encrypted bytes
   │  5. decrypt locally with your result_key   │
   ▼
result.paths_returns / paths_prices + diagnostics
```

## What's alpha vs production

The alpha endpoint runs on an **L4 GPU** in a regular VM, NOT a confidential VM.
Currently located in `europe-west4-a` (us-central1 L4 capacity is in stockout
at the time of writing; we'll re-region when GCP capacity opens up).
The model output is real — same Sablier flow model that ships in production.
The encryption is real. The attestation handshake is real (structure-only).

**What's not yet production-grade**:

- **GPU memory isn't encrypted** (no SEV-SNP + H100 CC mode). Alpha customers
  trust Sablier-controlled GCP infrastructure for the GPU side.
- **Attestation cryptography is real for AMD SEV-SNP** but the alpha doesn't
  run on SEV-SNP yet — verification runs in `fake-for-dev` mode.
- **The alpha endpoint is a single SPOT VM**. It may be preempted; jobs in
  flight are dropped, and the customer SDK surfaces this as a
  ``TransportError``. We've seen typical uptime of several hours per VM.
  Re-fetch the endpoint URL + cert from GCS to point at the recreated
  instance. Production swaps SPOT for on-demand + auto-restart.

### About on-demand availability

GCP L4 on-demand capacity is intermittent across us-central1 and europe-west4
(common for GPU SKUs on busy days). Until that loosens, the alpha runs on
SPOT, which trades availability guarantees for ~4x lower cost. The
``$SABLIER_FLOW_PINNED_IMAGE_DIGEST`` and the cert pinning both work
identically across SPOT and on-demand — only the underlying compute fleet
changes.

Once GCP approves our H100 quota, the alpha endpoint swaps to A3 H100 CC mode.
**Your SDK code does not change** — just the `attestation_mode` switches from
`"fake-for-dev"` to `"production"` and `verify=` points at the real cert.

## What you can do today

✅ Validate that the SDK fits your existing workflow
✅ Validate that your backtest engine consumes the synthetic data cleanly
✅ See whether the overfit-verdict bands match your intuition on
   known-overfit vs known-robust strategies
✅ Stress-test the integration against your real proprietary data
✅ Compare verdicts across multiple backtest engines (pandas, backtrader,
   vectorbt, LEAN) on the same flow-model output

## What you cannot do today

❌ Run regulated-data jobs that require attested GPU memory encryption
   — wait for A3 H100 CC mode
❌ Expect 99.9% uptime — alpha is a single spot VM
❌ `pip install sablier-flow` — install from git until PyPI publish

---

## Getting help

- Sablier: <https://sablier.ai>
- Email: `team@sablier.it`
