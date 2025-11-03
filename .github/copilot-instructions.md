<!-- .github/copilot-instructions.md: Guidance for AI coding agents working on this repository -->

Short goal
- Help the maintainers build a simple ML pipeline that loads the Wisconsin Breast Cancer dataset (contents in `docs/`), prepares features, and trains a binary classifier. Keep changes small, testable, and documented.

Quick repository tour
- `src/application/prepare_data.py` — current Python stub; primary place to implement data-loading and preprocessing.
- `src/application/MyFirstNotebook.ipynb` — exploratory notebook; use as a reference for demos and experiments. Notebook cells are unexecuted in repo.
- `docs/breast+cancer+wisconsin+diagnostic.zip` — dataset archive. Expect to extract a CSV or similar and place a copy under a `data/` folder for code to consume.

What to do first (high-value tasks)
- Implement a robust data loader in `src/application/prepare_data.py` that: reads the dataset from `docs/` (or `data/` if added), performs minimal cleaning (missing values, type casts), and returns a pandas DataFrame. Add a small unit test in `tests/` verifying loader output shape and column names.
- Make the notebook reproducible: add a cell that imports the new loader and runs the preprocessing pipeline; mark seed values for deterministic behavior.

Project conventions & patterns
- Keep ML code in `src/application/` and tests in `tests/`.
- Prefer small, single-purpose functions (loader, feature_engineer, split_data, train_model) so the notebook can import and compose them.
- Use pandas for dataframes and scikit-learn for modeling unless a clear reason to use alternatives is found.

Build / test / debug notes (discovered)
- No explicit package manager files found. Use the repository Python environment (create a venv or use the provided `env/` if it contains a virtualenv). Typical dev commands:
  - Install deps: `pip install pandas scikit-learn pytest` (adjust in a requirements.txt if added).
  - Run tests: `pytest -q`.
  - Run the notebook locally with Jupyter: `jupyter notebook src/application/MyFirstNotebook.ipynb`.

Integration points & external dependencies
- The primary external dependency is the dataset in `docs/` (zip archive). No external APIs or services detected.

When editing
- Make minimal, easily reviewed commits. Add unit tests for behavior changes. If adding new files (data or packages), update README or add `requirements.txt`.

Examples from this codebase
- Data loader: implement a function like `def load_breast_cancer(path: str) -> pd.DataFrame:` placed in `src/application/prepare_data.py` and imported from the notebook.
- Notebook usage: add an example cell to `MyFirstNotebook.ipynb` that calls the loader, prints DataFrame.shape, and shows head(5).

Constraints & gotchas
- The repo currently lacks CI configuration and dependency manifests. Avoid making assumptions about available packages—add `requirements.txt` if you introduce new dependencies.
- Files under `env/` may contain a virtual environment; respect it when describing commands.

If you need clarification
- Ask which dependency management approach (requirements.txt, pipenv, conda) the maintainers prefer and whether dataset should be committed under `data/` or downloaded at runtime.

Feedback request
- After applying changes, ask the human reviewer to confirm paths and preferred packaging before expanding the pipeline.
