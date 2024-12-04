import winston from 'winston';

// Configure logging
const logger = winston.createLogger({
    level: 'info',
    format: winston.format.json(),
    transports: [
        new winston.transports.Console({
            format: winston.format.simple(),
        }),
    ],
});

/**
 * Decorator for agent functions
 * @param {Object} config - Configuration object
 * @param {string} config.category - Function category
 * @param {string} config.description - Function description
 * @param {string[]} config.agentTriggers - List of agent triggers
 * @returns {Function} Decorated function
 */
export function AgentFunction(config) {
    return function(target) {
        return async function(...args) {
            logger.info(`Executing ${target.name}`);
            try {
                const result = await target.apply(this, args);
                return result;
            } catch (error) {
                logger.error(`Error in ${target.name}: ${error.message}`);
                throw error;
            }
        };
    };
}

/**
 * Decorator for workflows
 * @param {Function} target - Target function
 * @returns {Function} Decorated function
 */
export function workflow(target) {
    return async function(...args) {
        logger.info(`Starting workflow: ${target.name}`);
        try {
            const result = await target.apply(this, args);
            logger.info(`Completed workflow: ${target.name}`);
            return result;
        } catch (error) {
            logger.error(`Workflow ${target.name} failed: ${error.message}`);
            throw error;
        }
    };
}

/**
 * Base class for validation errors
 */
export class ValidationError extends Error {
    constructor(message) {
        super(message);
        this.name = 'ValidationError';
    }
}
