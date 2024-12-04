import fetch from 'node-fetch';

/**
 * Client for calling Python agent functions from JavaScript
 */
export class PythonFunctionClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }

    /**
     * List all available Python functions
     * @returns {Promise<Array>} List of available functions
     */
    async listFunctions() {
        const response = await fetch(`${this.baseUrl}/available_functions`);
        const data = await response.json();
        return data.functions;
    }

    /**
     * Call a Python function
     * @param {string} functionName - Full name of the Python function
     * @param {Array} args - Positional arguments
     * @param {Object} kwargs - Keyword arguments
     * @returns {Promise<any>} Function result
     */
    async callFunction(functionName, args = [], kwargs = {}) {
        const response = await fetch(`${this.baseUrl}/execute`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                function_name: functionName,
                args,
                kwargs,
            }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(`Python function error: ${error.detail}`);
        }

        const data = await response.json();
        return data.result;
    }
}

/**
 * Decorator that creates a JavaScript wrapper for a Python function
 * @param {string} pythonFunctionName - Full name of the Python function
 * @returns {Function} Decorated function
 */
export function PythonFunction(pythonFunctionName) {
    return function(target) {
        return async function(...args) {
            const client = new PythonFunctionClient();
            return await client.callFunction(pythonFunctionName, args);
        };
    };
}

// Example usage:
if (process.argv[1] === new URL(import.meta.url).pathname) {
    // Create a function that calls a Python function
    const analyzeSentiment = PythonFunction('examples.agent_triggered_example.analyze_sentiment')(
        async function analyzeSentiment(text) {
            return text; // This implementation is replaced by the decorator
        }
    );

    // Test the function
    async function test() {
        try {
            const client = new PythonFunctionClient();
            
            console.log('Available Python functions:');
            const functions = await client.listFunctions();
            console.log(JSON.stringify(functions, null, 2));
            
            console.log('\nTesting sentiment analysis:');
            const result = await analyzeSentiment(
                'This is a great example of Python-JavaScript interop!'
            );
            console.log('Result:', JSON.stringify(result, null, 2));
        } catch (error) {
            console.error('Error:', error.message);
        }
    }

    test();
}
