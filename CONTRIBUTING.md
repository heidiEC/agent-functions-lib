# Contributing to Agent Functions Library

First off, thank you for considering contributing to Agent Functions Library! It's people like you that make this library a great tool for AI-driven development.

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* Use a clear and descriptive title
* Describe the exact steps which reproduce the problem
* Provide specific examples to demonstrate the steps
* Describe the behavior you observed after following the steps
* Explain which behavior you expected to see instead and why
* Include code samples and stack traces if applicable

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* A clear and descriptive title
* A detailed description of the proposed functionality
* Explain why this enhancement would be useful to most Agent Functions Library users
* List some examples of how the enhancement would be used
* If applicable, include code snippets demonstrating the enhancement

### Pull Requests

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code lints

## Development Setup

1. Clone your fork of the repository
```bash
git clone https://github.com/YOUR_USERNAME/agent-functions-lib.git
```

2. Install Python dependencies
```bash
pip install -e ".[dev]"
```

3. Install Node.js dependencies
```bash
npm install
```

4. Run the test suites
```bash
# Python tests
pytest

# JavaScript tests
npm test
```

## Style Guides

### Git Commit Messages

* Use the present tense ("add feature" not "added feature")
* Use the imperative mood ("move cursor to..." not "moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line
* Consider starting the commit message with an applicable emoji:
    * 🎨 `:art:` when improving the format/structure of the code
    * 🐎 `:racehorse:` when improving performance
    * 📝 `:memo:` when writing docs
    * 🐛 `:bug:` when fixing a bug
    * 🔥 `:fire:` when removing code or files
    * ✅ `:white_check_mark:` when adding tests

### Python Style Guide

* Follow PEP 8
* Use type hints
* Use docstrings in Google format
* Maximum line length is 88 characters (compatible with black)
* Use f-strings for string formatting

Example:
```python
from typing import Dict, Any

def process_data(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process input data and return results.
    
    Args:
        input_data: Dictionary containing input parameters
        
    Returns:
        Processed data dictionary
        
    Raises:
        ValueError: If input_data is invalid
    """
    return {"result": "processed"}
```

### JavaScript Style Guide

* Use ES6+ features
* Use async/await over promises
* Use TypeScript-style JSDoc comments
* Maximum line length is 100 characters
* Use const and let, avoid var

Example:
```javascript
/**
 * Process input data and return results
 * @param {Object} inputData - Input parameters
 * @returns {Promise<Object>} Processed data
 * @throws {Error} If inputData is invalid
 */
async function processData(inputData) {
    return { result: 'processed' };
}
```

## Testing Guidelines

1. Write tests for all new features
2. Maintain test coverage above 80%
3. Use descriptive test names that explain the expected behavior
4. Structure tests using the Arrange-Act-Assert pattern
5. Mock external dependencies appropriately

### Python Testing Example
```python
def test_sentiment_analysis_returns_expected_score():
    # Arrange
    text = "This is a great example!"
    
    # Act
    result = analyze_sentiment(text)
    
    # Assert
    assert "score" in result
    assert isinstance(result["score"], float)
    assert 0 <= result["score"] <= 1
```

### JavaScript Testing Example
```javascript
describe('analyzeSentiment', () => {
    it('returns expected score for positive text', async () => {
        // Arrange
        const text = 'This is a great example!';
        
        // Act
        const result = await analyzeSentiment(text);
        
        // Assert
        expect(result).toHaveProperty('score');
        expect(typeof result.score).toBe('number');
        expect(result.score).toBeGreaterThanOrEqual(0);
        expect(result.score).toBeLessThanOrEqual(1);
    });
});
```

## Documentation Guidelines

1. Keep README.md up to date
2. Document all public APIs
3. Include examples in documentation
4. Update CHANGELOG.md for all notable changes
5. Add JSDoc or Python docstrings for all public functions

## Questions?

Feel free to open an issue tagged as a question if you need help or clarification.

Thank you for contributing to Agent Functions Library! 🚀
