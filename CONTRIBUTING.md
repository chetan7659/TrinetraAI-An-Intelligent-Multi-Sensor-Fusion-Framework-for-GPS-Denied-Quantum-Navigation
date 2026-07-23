# Contributing to Trinetra-AI

First off, thank you for considering contributing to Trinetra-AI!
This document provides guidelines to ensure code quality and consistency across the project.

## Branch Strategy
- `main` is the primary branch. It should always be stable and deployable.
- Feature branches should be created from `main` and named descriptively: `feature/dataset-adapter`, `bugfix/kalman-filter`, `docs/architecture-update`.
- Once a feature is complete, open a Pull Request (PR) targeting `main`.

## Commit Message Convention
We use Conventional Commits. Each commit message should follow this format:
```
<type>(<scope>): <subject>
```
Types:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools and libraries such as documentation generation

## Code Style
- **Formatting**: We use `ruff` to enforce PEP 8 standards.
- **Naming Conventions**:
  - Packages and modules: `snake_case`
  - Classes: `PascalCase`
  - Variables, functions, methods: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
- **Architecture**: Ensure that any changes respect the Clean Architecture boundaries (Dependencies always point inward toward Domain layer).
- **Docstrings**: We use Google-style docstrings for all classes and public methods.

## Testing Requirements
- All new features and bug fixes must include unit tests.
- We use `pytest` for testing.
- Place tests in the `tests/` directory following the `tests/unit/`, `tests/integration/` structure depending on test scope.
- To run tests: `pytest tests/`

## Pre-commit Hooks
Please install and run pre-commit hooks before committing:
```bash
pre-commit install
pre-commit run --all-files
```
