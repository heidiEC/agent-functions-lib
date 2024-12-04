/**
 * Core mathematical operations optimized for agent use.
 */

const { AgentFunction } = require('../core');
const math = require('mathjs');

/**
 * Add two numbers or vectors together
 * @param {number|Array} a First number or vector
 * @param {number|Array} b Second number or vector
 * @returns {number|Array} Sum of inputs
 */
const add = AgentFunction({
    category: 'math.basic',
    description: 'Add two numbers or vectors together',
    agentTriggers: ['addition_requested', 'sum_needed', 'combine_numbers'],
    examples: [
        { inputs: { a: 5, b: 3 }, output: 8 },
        { inputs: { a: [1, 2], b: [3, 4] }, output: [4, 6] }
    ]
})(
    async function(a, b) {
        return math.add(a, b);
    }
);

/**
 * Calculate basic statistics of a numeric sequence
 * @param {Array<number>} values List of numbers
 * @returns {Object} Dictionary with mean and standard deviation
 */
const calculateStatistics = AgentFunction({
    category: 'math.statistics',
    description: 'Calculate mean and standard deviation of a numeric sequence',
    agentTriggers: ['stats_needed', 'distribution_analysis'],
    examples: [
        { 
            inputs: { values: [1, 2, 3, 4, 5] },
            output: { mean: 3.0, std: 1.5811388300841898 }
        }
    ]
})(
    async function(values) {
        return {
            mean: math.mean(values),
            std: math.std(values)
        };
    }
);

/**
 * Multiply two matrices or perform scalar multiplication
 * @param {Array<Array<number>>} a First matrix
 * @param {Array<Array<number>>} b Second matrix
 * @returns {Array<Array<number>>} Result of matrix multiplication
 */
const matrixMultiply = AgentFunction({
    category: 'math.linear_algebra',
    description: 'Multiply two matrices or perform scalar multiplication',
    agentTriggers: ['matrix_multiplication', 'scale_matrix'],
    examples: [
        {
            inputs: {
                a: [[1, 2], [3, 4]],
                b: [[5, 6], [7, 8]]
            },
            output: [[19, 22], [43, 50]]
        }
    ]
})(
    async function(a, b) {
        return math.multiply(a, b);
    }
);

/**
 * Perform correlation analysis between two vectors
 * @param {Array<number>} x First vector
 * @param {Array<number>} y Second vector
 * @returns {Object} Dictionary with correlation coefficient and p-value
 */
const correlationAnalysis = AgentFunction({
    category: 'math.statistics',
    description: 'Perform correlation analysis between two vectors',
    agentTriggers: ['correlation_needed', 'relationship_analysis'],
    examples: [
        {
            inputs: {
                x: [1, 2, 3, 4, 5],
                y: [2, 4, 5, 4, 5]
            },
            output: {
                correlation: 0.8164965809277261,
                pValue: 0.09186468873368339
            }
        }
    ]
})(
    async function(x, y) {
        const n = x.length;
        const { mean, std } = math;
        
        // Calculate means
        const meanX = mean(x);
        const meanY = mean(y);
        
        // Calculate correlation coefficient
        let numerator = 0;
        for (let i = 0; i < n; i++) {
            numerator += (x[i] - meanX) * (y[i] - meanY);
        }
        const denominator = std(x) * std(y) * (n - 1);
        const correlation = numerator / denominator;
        
        // Calculate t-statistic for p-value
        const t = correlation * Math.sqrt((n-2)/(1-correlation*correlation));
        const pValue = 2 * (1 - math.erf(Math.abs(t)/Math.sqrt(2)));
        
        return {
            correlation,
            pValue
        };
    }
);

module.exports = {
    add,
    calculateStatistics,
    matrixMultiply,
    correlationAnalysis
};
