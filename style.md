# Coding Style Guide (`style.md`)

This document defines the code style, linting rules, formatting standards, and architectural conventions for the `multimodal-uncertainty` repository.

---

## General Principles

1. **PEP 8 Compliance**: Follow standard Python style guidelines.
2. **Formatter**: All code MUST be formatted using `black` with a **100-character line length limit**.
3. **Linter**: `ruff` is the standard linter. No unused imports (`F401`), undefined variables (`F821`), or trailing whitespace.
4. **Type Annotations**: All public functions, methods, and class attributes MUST include explicit type hints (`typing` or Python 3.10+ native annotations like `str | None`).
5. **Docstrings**: Public classes, methods, and functions MUST include Google-style docstrings.

---

## Code Formatting Commands

```bash
# Auto-format codebase
black --line-length 100 src tests utils

# Run linter
ruff check src tests utils
```

---

## Pydantic v2 Guidelines

- Use Pydantic v2 `BaseModel` and `Field` declarations.
- Prefer `model_dump()` over deprecated `dict()`.
- Use `model_validate_json()` for JSON parsing.

---

## Error Handling & Fallbacks

- Never swallow exceptions silently. Log or wrap exceptions in explicit `ImportError` or `ValueError` messages.
- Always provide mock/fallback options for engine callers when external APIs (e.g. Google GenAI or local servers) are unreachable during local unit testing.
