import { AgentFunction, workflow } from '../../src/js/core.js';

/**
 * Analyze text sentiment
 * @param {string} text - Input text to analyze
 * @returns {Object} Sentiment scores
 */
const analyzeSentiment = AgentFunction({
    category: 'nlp',
    description: 'Analyze text sentiment',
    agentTriggers: ['sentiment_analysis_requested', 'emotion_detection_needed']
})(
    function analyzeSentiment(text) {
        // Simplified sentiment scoring
        const positiveWords = new Set(['good', 'great', 'excellent', 'happy', 'positive']);
        const negativeWords = new Set(['bad', 'poor', 'terrible', 'sad', 'negative']);
        
        const words = text.toLowerCase().split(/\s+/);
        const posScore = words.filter(word => positiveWords.has(word)).length / words.length;
        const negScore = words.filter(word => negativeWords.has(word)).length / words.length;
        
        return {
            positive_score: posScore,
            negative_score: negScore,
            neutral_score: 1 - (posScore + negScore)
        };
    }
);

/**
 * Extract entities from text
 * @param {string} text - Input text for entity extraction
 * @returns {Object} Extracted entities
 */
const extractEntities = AgentFunction({
    category: 'data_transform',
    description: 'Extract key entities from text',
    agentTriggers: ['entity_extraction_needed', 'key_concept_identification']
})(
    function extractEntities(text) {
        const words = text.split(/\s+/);
        const capitalized = words.filter(word => /^[A-Z]/.test(word));
        
        return {
            potential_entities: capitalized,
            word_count: words.length
        };
    }
);

/**
 * Process text with analytics workflow
 * @param {Object} event - Lambda event
 * @returns {Object} Processing results
 */
const processTextWithAnalytics = workflow(
    async function processTextWithAnalytics(event) {
        const text = event.text || '';
        const trigger = event.trigger || '';
        
        const results = {};
        
        if (trigger.toLowerCase().includes('sentiment')) {
            results.sentiment = await analyzeSentiment(text);
        }
        
        if (trigger.toLowerCase().includes('entity')) {
            results.entities = await extractEntities(text);
        }
        
        return results;
    }
);

/**
 * AWS Lambda handler
 * @param {Object} event - Lambda event
 * @param {Object} context - Lambda context
 * @returns {Object} API Gateway response
 */
export async function handler(event, context) {
    try {
        const body = typeof event.body === 'string' ? JSON.parse(event.body) : (event.body || {});
        const results = await processTextWithAnalytics(body);
        
        return {
            statusCode: 200,
            body: JSON.stringify(results),
            headers: {
                'Content-Type': 'application/json'
            }
        };
    } catch (error) {
        return {
            statusCode: 500,
            body: JSON.stringify({ error: error.message }),
            headers: {
                'Content-Type': 'application/json'
            }
        };
    }
}

// Local testing
if (process.argv[1] === new URL(import.meta.url).pathname) {
    // Simulate an agent triggering sentiment analysis
    const testEvent = {
        body: {
            text: 'This is a great example of agent-triggered functions! However, some parts need improvement.',
            trigger: 'sentiment_analysis_requested'
        }
    };
    
    handler(testEvent, {})
        .then(result => console.log('Test result:', JSON.stringify(result, null, 2)))
        .catch(error => console.error('Test error:', error));
}
