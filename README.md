# Agent Functions Library

## Overview
A modular, extensible library of functions designed for agent workflows, providing a flexible framework for creating and composing agent-based tasks.

## Features
- Modular function design
- Easy extensibility
- Workflow composition
- Type-safe function interfaces

## Installation
```bash
pip install agent-functions
```

## Quick Start
```python
from agent_functions import AgentFunction, workflow

@AgentFunction(category="data_transform")
def clean_text(input_text: str) -> str:
    return input_text.strip().lower()

@workflow
def process_data(text: str):
    cleaned = clean_text(text)
    # Additional processing steps
    return cleaned
```

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
MIT License
