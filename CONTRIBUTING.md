# Contributing

This repository is intentionally narrow in scope.

Please keep contributions aligned with the core study design:

- evaluate **GPT-5.2 only**
- compare the four public reasoning settings: `none`, `low`, `medium`, `high`
- use **GPT-4.1** as the grader unless adding a clearly optional and documented fallback
- keep the codebase readable, explicit, and easy to reproduce

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

## Pull request expectations

- no dead code
- no provider-specific abstractions unless they are genuinely needed
- no stale generated results in commits
- keep docs and config examples in sync with code changes
