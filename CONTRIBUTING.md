# Contributing to `multimodal-uncertainty`

Thank you for your interest in contributing to `multimodal-uncertainty`! We welcome bug fixes, documentation improvements, new engine adapters, and scientific calibration enhancements.

---

## Development Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/dabrign/uncertanty.git
   cd multimodal-uncertainty
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Package in Editable Mode with Dev Dependencies**:
   ```bash
   pip install -e ".[dev,all]"
   ```

---

## Running Tests & Code Quality Checks

We use `pytest` for unit testing and `ruff` / `black` for formatting.

```bash
# Run unit test suite
pytest

# Check linting rules
ruff check src tests

# Check code formatting
black --check src tests
```

---

## Workflow for Pull Requests

1. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Commit Changes**:
   Write concise, descriptive commit messages adhering to Conventional Commits:
   - `feat: add Anthropic logprob engine`
   - `fix: handle edge case in JSON token isolation`
   - `docs: update quickstart example in README`

3. **Open a Pull Request**:
   Ensure all automated CI checks pass before requesting review.

---

## Code of Conduct & Licensing

By contributing to this project, you agree that your contributions will be licensed under the project's [Apache 2.0 License](LICENSE).
