# Contributing to AgentDS

Thank you for your interest in contributing to AgentDS! This document provides guidelines and information for contributors.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Making Contributions](#making-contributions)
5. [Code Style](#code-style)
6. [Testing](#testing)
7. [Documentation](#documentation)
8. [Pull Request Process](#pull-request-process)

---

## Code of Conduct

This project follows a standard code of conduct. Please be respectful and inclusive in all interactions.

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Redis (for testing)
- Docker (optional, for containerized testing)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/agentds-v2.git
   cd agentds-v2
   ```
3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/mlvpatel/agentds-v2.git
   ```

---

## Development Setup

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

### Install Development Dependencies

```bash
pip install -e ".[dev]"
```

### Setup Pre-commit Hooks

```bash
pre-commit install
```

### Configure Environment

```bash
cp .env.example .env
# Edit .env with your development API keys
```

---

## Making Contributions

### Types of Contributions

We welcome:

- **Bug fixes**: Fix issues and improve stability
- **New features**: Add new agents, integrations, or capabilities
- **Documentation**: Improve docs, add examples, fix typos
- **Tests**: Add test coverage
- **Performance**: Optimize code and reduce resource usage

### Branching Strategy

- `main`: Stable release branch
- `develop`: Development branch
- Feature branches: `feature/your-feature-name`
- Bug fix branches: `fix/issue-description`

### Creating a Branch

```bash
# Update from upstream
git fetch upstream
git checkout develop
git merge upstream/develop

# Create feature branch
git checkout -b feature/my-new-feature
```

---

## Code Style

### Python Style Guide

We follow PEP 8 with these tools:

- **Ruff**: For linting and formatting
- **Black**: For code formatting (via Ruff)
- **isort**: For import sorting (via Ruff)
- **mypy**: For type checking

### Run Linting

```bash
# Run all checks
ruff check .

# Fix auto-fixable issues
ruff check --fix .

# Type checking
mypy agentds/
```

### Code Standards

1. **Type hints**: All functions must have type hints
2. **Docstrings**: Use Google-style docstrings
3. **Max line length**: 88 characters (Black default)
4. **Import order**: stdlib, third-party, local (isort)
5. **No emojis**: Use text indicators [OK], [ERROR], [WARN]

### Example Function

```python
def process_data(
    data: pl.DataFrame,
    columns: list[str],
    threshold: float = 0.5,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """
    Process data and return results.

    Args:
        data: Input DataFrame to process
        columns: List of column names to include
        threshold: Quality threshold (0.0 to 1.0)

    Returns:
        Tuple of processed DataFrame and metadata dict

    Raises:
        ValueError: If threshold is out of range
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")

    # Implementation...
    return processed_data, metadata
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=agentds --cov-report=html

# Run specific test file
pytest tests/test_agents.py -v

# Run specific test
pytest tests/test_agents.py::TestDataLoaderAgent::test_init -v
```

### Test Categories

```bash
# Run only unit tests
pytest tests/ -m unit

# Run only integration tests
pytest tests/ -m integration

# Skip slow tests
pytest tests/ -m "not slow"
```

### Writing Tests

1. Place tests in `tests/` directory
2. Name files `test_*.py`
3. Name test functions `test_*`
4. Use pytest fixtures for setup
5. Mock external services (LLM APIs, databases)

### Example Test

```python
import pytest
from unittest.mock import MagicMock

from agentds.agents import DataLoaderAgent
from agentds.core.config import Settings


@pytest.fixture
def settings() -> Settings:
    """Create test settings."""
    return Settings(debug=True)


@pytest.fixture
def mock_llm_gateway() -> MagicMock:
    """Create mock LLM gateway."""
    gateway = MagicMock()
    gateway.complete.return_value = MagicMock(content="Test response")
    return gateway


class TestDataLoaderAgent:
    """Tests for DataLoaderAgent."""

    def test_init(self, settings: Settings):
        """Test agent initialization."""
        agent = DataLoaderAgent(settings=settings)
        assert agent.name == "DataLoaderAgent"
        assert agent.phase == "build"

    def test_load_csv(self, settings: Settings, tmp_path):
        """Test loading CSV file."""
        # Create test file
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n")

        agent = DataLoaderAgent(settings=settings)
        df = agent._load_file(csv_file, "csv", {})

        assert df.height == 2
        assert df.width == 3
```

---

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int = 10) -> bool:
    """
    Brief description of function.

    Longer description if needed. Can span multiple lines
    and include additional context.

    Args:
        param1: Description of first parameter
        param2: Description of second parameter with default

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is empty
        TypeError: When param2 is not an integer

    Example:
        >>> example_function("hello", 5)
        True
    """
```

### Documentation Files

- `README.md`: Project overview
- `docs/ARCHITECTURE.md`: Technical architecture
- `docs/USER_MANUAL.md`: User guide
- `docs/API_REFERENCE.md`: API documentation
- `docs/N8N_GUIDE.md`: n8n integration guide

---

## Pull Request Process

### Before Submitting

1. **Update your branch**:
   ```bash
   git fetch upstream
   git rebase upstream/develop
   ```

2. **Run all checks**:
   ```bash
   ruff check .
   mypy agentds/
   pytest tests/
   ```

3. **Update documentation** if needed

4. **Add tests** for new functionality

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added
- [ ] Coverage maintained/improved

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
```

### Review Process

1. Create PR against `develop` branch
2. Wait for CI checks to pass
3. Address reviewer feedback
4. Maintainer will merge when approved

---

## Release Process

Releases are managed by maintainers:

1. Features merged to `develop`
2. Release branch created from `develop`
3. Version bumped and changelog updated
4. PR to `main` after testing
5. Tagged and published to PyPI

---

## Getting Help

- **Issues**: Open a GitHub issue for bugs/features
- **Discussions**: Use GitHub Discussions for questions
- **Email**: malav.patel203@gmail.com

---

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in documentation

---

Thank you for contributing to AgentDS!

*Author: Malav Patel*
