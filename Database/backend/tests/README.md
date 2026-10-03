# Test boundaries

From the repository root, install `Database/backend/requirements-test.txt` into
the development environment, then run:

```powershell
# Bender / PowerShell, E:\LoRA Project
Set-Location 'E:\LoRA Project'
& .\.venv\Scripts\python.exe -m pytest Database/backend/tests -q -ra
```

The default suite uses synthetic data and temporary SQLite databases. It does
not need the personal model library, a running ComfyUI, a GPU, or PyTorch. The
three real tensor extraction tests in `test_unet_block_extractor.py` explicitly
skip as one module when PyTorch is absent. The PEFT namespace/pairing tests use a
private tensor double; they do not claim to validate PyTorch numerical work and
must never install a fake `torch` in the process module registry.

The `CI` workflow runs this suite plus the backup/questionnaire checks, and a
separate UI job runs `npm ci`, lint, tests and build. Those gates characterise
the code; they are not evidence of improved generated images or loader behavior
inside a real ComfyUI session.

For a real tensor check, manually run `CI` with **real_tensors** enabled. That
separate job installs the same CPU PyTorch version pinned by the existing Docker
runtime, requires a real tensor operation before running the whole backend
suite, and fails if PyTorch cannot import. It needs no private models. Ordinary
pull requests do not download that large optional runtime.

If Windows sandbox permissions prevent pytest from creating its default temp
directory, add `--basetemp=.local/pytest-UNIQUE-RUN-NAME` using a new empty name
for each run. Never point `--basetemp` at a directory containing useful files:
pytest clears it.
