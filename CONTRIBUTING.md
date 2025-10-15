# Contributing to Ascend-Pangu-Infer

Thank you for your interest in contributing to Ascend-Pangu-Infer! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. Check if the issue already exists in the [Issues](https://github.com/VivianZan/Ascend-Pangu-Infer/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs. actual behavior
   - Environment details (OS, Python version, hardware)

### Submitting Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/VivianZan/Ascend-Pangu-Infer.git
   cd Ascend-Pangu-Infer
   ```

2. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the code style guidelines
   - Add tests for new features
   - Update documentation as needed

4. **Test your changes**
   ```bash
   pytest tests/
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your branch
   - Provide a clear description of changes

## Code Style Guidelines

### Python Code

- Follow PEP 8 style guide
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use docstrings for all public functions and classes

Example:
```python
def generate_text(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_length: int = 128,
) -> torch.Tensor:
    """Generate text using the model.
    
    Args:
        model: The language model
        input_ids: Input token IDs
        max_length: Maximum generation length
        
    Returns:
        Generated token IDs
    """
    # Implementation
    pass
```

### Documentation

- Use clear, concise language
- Include code examples
- Update README.md for significant changes
- Add docstrings to new functions/classes

### Testing

- Add unit tests for new features
- Ensure existing tests pass
- Aim for >80% code coverage
- Use pytest for testing

Example test:
```python
def test_model_generation():
    """Test text generation."""
    config = PanguConfig()
    model = PanguModel(config)
    input_ids = torch.randint(0, 1000, (1, 10))
    
    output = model.generate(input_ids, max_length=20)
    
    assert output.shape[0] == 1
    assert output.shape[1] <= 20
```

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/VivianZan/Ascend-Pangu-Infer.git
   cd Ascend-Pangu-Infer
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**
   ```bash
   pip install -e .
   pip install pytest black flake8
   ```

4. **Run tests**
   ```bash
   pytest tests/ -v
   ```

5. **Format code**
   ```bash
   black pangu_infer/
   flake8 pangu_infer/
   ```

## Project Structure

```
Ascend-Pangu-Infer/
├── pangu_infer/          # Main package
│   ├── models/           # Model implementations
│   ├── quantization/     # Quantization module
│   ├── pruning/          # Pruning module
│   ├── speculative/      # Speculative inference
│   └── utils/            # Utilities
├── examples/             # Example scripts
├── tests/                # Unit tests
├── docs/                 # Documentation
└── README.md
```

## Areas for Contribution

We welcome contributions in the following areas:

### High Priority

- [ ] Additional quantization methods (GPTQ, AWQ)
- [ ] More pruning strategies
- [ ] Performance benchmarks
- [ ] Model conversion utilities
- [ ] Ascend NPU optimizations

### Medium Priority

- [ ] Support for more model architectures
- [ ] Distributed inference
- [ ] Mixed precision training
- [ ] Model serving utilities
- [ ] Visualization tools

### Documentation

- [ ] Tutorials and guides
- [ ] API documentation
- [ ] Performance tuning guide
- [ ] Troubleshooting guide
- [ ] Use case examples

## Code Review Process

1. All submissions require review
2. Maintainers will review PRs within 1-2 weeks
3. Address reviewer feedback promptly
4. Once approved, changes will be merged

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## Questions?

If you have questions:
- Open an issue with the "question" label
- Join our community discussions
- Contact the maintainers

Thank you for contributing to Ascend-Pangu-Infer!
