/**
 * Memory operations for storing and retrieving agent data.
 */

const { AgentFunction } = require('../core');
const fs = require('fs').promises;
const path = require('path');

class MemoryStore {
    /**
     * Simple key-value store with persistence
     * @param {string} storagePath Path to store data
     */
    constructor(storagePath = '.agent_memory') {
        this.storagePath = storagePath;
        this.ensureDirectory();
    }

    /**
     * Ensure storage directory exists
     */
    async ensureDirectory() {
        try {
            await fs.mkdir(this.storagePath, { recursive: true });
        } catch (error) {
            if (error.code !== 'EEXIST') throw error;
        }
    }

    /**
     * Get full path for a key
     * @param {string} key Storage key
     * @returns {string} Full file path
     */
    getPath(key) {
        return path.join(this.storagePath, `${key}.json`);
    }

    /**
     * Save value with metadata
     * @param {string} key Storage key
     * @param {*} value Value to store
     * @param {Object} metadata Optional metadata
     */
    async save(key, value, metadata = {}) {
        const data = {
            value,
            metadata,
            timestamp: new Date().toISOString()
        };
        await fs.writeFile(
            this.getPath(key),
            JSON.stringify(data, null, 2)
        );
    }

    /**
     * Load stored value
     * @param {string} key Storage key
     * @returns {Object|null} Stored data or null if not found
     */
    async load(key) {
        try {
            const data = await fs.readFile(this.getPath(key), 'utf8');
            return JSON.parse(data);
        } catch (error) {
            if (error.code === 'ENOENT') return null;
            throw error;
        }
    }

    /**
     * Delete stored value
     * @param {string} key Storage key
     * @returns {boolean} True if deleted, false if not found
     */
    async delete(key) {
        try {
            await fs.unlink(this.getPath(key));
            return true;
        } catch (error) {
            if (error.code === 'ENOENT') return false;
            throw error;
        }
    }

    /**
     * List all stored keys
     * @returns {Array<string>} List of keys
     */
    async listKeys() {
        const files = await fs.readdir(this.storagePath);
        return files
            .filter(f => f.endsWith('.json'))
            .map(f => f.slice(0, -5));
    }
}

// Initialize global memory store
const memoryStore = new MemoryStore();

/**
 * Store a value with optional metadata
 * @param {string} key Storage key
 * @param {*} value Value to store
 * @param {Object} metadata Optional metadata
 * @returns {Promise<boolean>} True if successful
 */
const store = AgentFunction({
    category: 'memory.basic',
    description: 'Store a value with an associated key',
    agentTriggers: ['store_value', 'remember_data', 'save_for_later'],
    examples: [
        {
            inputs: {
                key: 'user_preference',
                value: { theme: 'dark', language: 'en' },
                metadata: { source: 'user_settings' }
            },
            output: true
        }
    ]
})(
    async function(key, value, metadata = {}) {
        try {
            await memoryStore.save(key, value, metadata);
            return true;
        } catch (error) {
            throw new Error(`Failed to store value: ${error.message}`);
        }
    }
);

/**
 * Retrieve a stored value
 * @param {string} key Storage key
 * @returns {Promise<Object|null>} Stored data or null if not found
 */
const retrieve = AgentFunction({
    category: 'memory.basic',
    description: 'Retrieve a previously stored value by key',
    agentTriggers: ['retrieve_value', 'recall_data', 'get_stored'],
    examples: [
        {
            inputs: { key: 'user_preference' },
            output: {
                value: { theme: 'dark', language: 'en' },
                metadata: { source: 'user_settings' },
                timestamp: '2024-01-20T10:30:00'
            }
        }
    ]
})(
    async function(key) {
        return await memoryStore.load(key);
    }
);

/**
 * Delete a stored value
 * @param {string} key Storage key
 * @returns {Promise<boolean>} True if deleted, false if not found
 */
const forget = AgentFunction({
    category: 'memory.basic',
    description: 'Delete a stored value by key',
    agentTriggers: ['forget_value', 'delete_stored', 'remove_data'],
    examples: [
        {
            inputs: { key: 'user_preference' },
            output: true
        }
    ]
})(
    async function(key) {
        return await memoryStore.delete(key);
    }
);

/**
 * List all stored keys
 * @returns {Promise<Array<string>>} List of keys
 */
const listStored = AgentFunction({
    category: 'memory.query',
    description: 'List all stored keys',
    agentTriggers: ['list_stored', 'show_memory', 'get_keys'],
    examples: [
        {
            inputs: {},
            output: ['user_preference', 'calculation_result', 'task_status']
        }
    ]
})(
    async function() {
        return await memoryStore.listKeys();
    }
);

module.exports = {
    store,
    retrieve,
    forget,
    listStored
};
