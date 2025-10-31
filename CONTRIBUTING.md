# Contributing to Drive-and-Vibe

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone <your-fork-url>`
3. Install dependencies: `make install-dev`
4. Set up pre-commit hooks: `make setup`

## Development Workflow

1. Create a branch for your feature: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run linting: `make lint`
4. Run tests: `make test`
5. Format code: `make format`
6. Commit your changes
7. Push to your fork
8. Create a pull request

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where possible
- Write docstrings for all functions and classes
- Keep functions focused and single-purpose
- Add comments for complex logic

## Testing

- Write tests for new features
- Ensure all tests pass: `make test`
- Aim for good test coverage
- Test edge cases and error conditions

## Pull Request Process

1. Update documentation if needed
2. Update CHANGELOG.md with your changes
3. Ensure all CI checks pass
4. Request review from maintainers
5. Address any feedback

## Reporting Issues

When reporting bugs or requesting features:

- Use the issue templates
- Provide a clear description
- Include steps to reproduce (for bugs)
- Add relevant code or data samples
- Specify your environment (OS, Python version, etc.)

## Questions?

Open an issue with the `question` label for help or clarification.
