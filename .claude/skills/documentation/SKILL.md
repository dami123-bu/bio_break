---
name: documentation
description: Write or update documentation for a module, function, or component
---

## Documentation Guidelines

### Module-level Docstrings
Every Python file should have a module docstring at the top:

```python
"""
<module_name>.py

<One sentence describing what this module does.>

<Optional: 2-3 sentences on key design decisions, usage context, or gotchas.>
"""
```

### Function Docstrings
Use Google style:

```
python
def function_name(param1: type, param2: type) -> return_type:
    """Short description.

    Args:
        param1: Description.
        param2: Description.

    Returns:
        Description of return value.

    Raises:
        ExceptionType: When this happens.
    """
```

### README Updates
When adding a new module or changing behavior, update `README.md`:
- Add the module to the architecture table if it's a new file
- Update the "Getting Started" or "Usage" section if the interface changed
- Note any new environment variables or dependencies

### What Not to Document
- Don't add comments explaining what the code does line-by-line — write clear code instead
- Don't document internal implementation details that are obvious from reading the code
- Don't add TODO comments — raise an open question in the plan skill instead

### Research-specific
- Document the threat model assumptions for any security-relevant module
- If a function behaves differently in attack vs. baseline mode, make that explicit in the docstring

### Changes
Read `.claude/skills/changes/SKILL.md` and update `CHANGES.md` accordingly
