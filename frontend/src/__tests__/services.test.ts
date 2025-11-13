import { describe, it, expect } from 'vitest';

describe('PredictionService', () => {
  it('exports predictionService', async () => {
    const { predictionService } = await import('../services/predictionService');
    expect(predictionService).toBeDefined();
    expect(predictionService.createPrediction).toBeDefined();
    expect(predictionService.getPrediction).toBeDefined();
    expect(predictionService.listPredictions).toBeDefined();
  });
});

describe('CampaignService', () => {
  it('exports campaignService', async () => {
    const { campaignService } = await import('../services/campaignService');
    expect(campaignService).toBeDefined();
  });
});

describe('ResultService', () => {
  it('exports resultService', async () => {
    const { resultService } = await import('../services/resultService');
    expect(resultService).toBeDefined();
  });
});
