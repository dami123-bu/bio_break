---
name: testing
description: Guidelines for writing tests in the Digital Ghost project
---

## Testing Guidelines

This project uses **pytest**. All test files go in a `tests/` directory at the project root.

### File Naming
- Unit tests: `tests/test_<module>.py` (e.g., `tests/test_tools.py`)
- Integration tests: `tests/integration/test_<feature>.py`

### What to Test

**For each module, cover:**
- Happy path (expected inputs and outputs)
- Edge cases and boundary conditions
- Error handling (invalid inputs, missing data, exceptions)

**Module-specific guidance:**

| Module | Key things to test |
|--------|-------------------|
| `ingest.py` | Documents loaded into ChromaDB correctly; duplicate handling |


### Test Structure

```python
import pytest

def test_something():
    # Arrange
    ...
    # Act
    ...
    # Assert
    ...
```

### Fixtures and Mocking
- Use `pytest` fixtures for shared setup (e.g., a fresh SQLite DB, a temp ChromaDB collection)
- Mock LLM API calls (Anthropic/OpenAI) — do not make real API calls in tests
- Use `tmp_path` fixture for temporary files and directories

### Security Research Context
- Tests for attack scenarios should be isolated and use mock/temp data stores
- Never use production ChromaDB collections or real compound data in tests
- Baseline (clean) and attacked states must be tested independently — do not share state between them

### Running Tests
```bash
pytest tests/
pytest tests/ -v          # verbose
pytest tests/test_tools.py  # single file
```