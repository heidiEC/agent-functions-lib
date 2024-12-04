import { extractEntitiesAndRelations } from './core_functions/cognitive_ops.js';

async function testTensorHandling() {
    try {
        const result = await extractEntitiesAndRelations("Tim Cook is the CEO of Apple in California. He announced the new iPhone.");
        console.log('Extraction result:', JSON.stringify(result, null, 2));
    } catch (error) {
        console.error('Error during testing:', error);
    }
}

testTensorHandling();
