
# AgentDojo - Project Instructions

## Project context
This repository is an academic research project about
Indirect Prompt Injection (IPI) in LLM-based agents.

The project uses AgentDojo as the experimental benchmark.
The research evaluates attack success, utility, and
reasoning/conditional logic corruption.

## Python environment
- Python 3.12.
- Dependencies are managed with uv.
- The project virtual environment is .venv.
- Use `uv run` for project commands.
- Use `uv pip install` for installing dependencies.
- Never modify the global Python environment.
- Do not recreate .venv without explicit authorization.

## Code modification rules
- Preserve existing experimental methodology.
- Do not change benchmark behavior unless necessary.
- Inspect callers, imports, and related definitions before
  changing function signatures.
- Avoid unrelated refactoring.
- Do not modify datasets, experiment outputs, or model
  configurations unless explicitly requested.

## Context and token efficiency
- Search for relevant symbols and imports before opening files.
- Read only files necessary to understand the issue.
- Do not recursively read the entire repository.
- Exclude .venv, .git, cache directories, datasets,
  model files, and experiment outputs from unnecessary analysis.
- Follow dependencies only as far as needed to understand
  the affected code.

## Validation
- Run Ruff on modified Python files.
- Run Pyright on modified Python files.
- Run relevant tests when available.
- Report commands executed and their actual results.
- Never claim a test passed if it was not executed.
