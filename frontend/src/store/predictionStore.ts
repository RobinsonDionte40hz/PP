import { create } from 'zustand';
import type { PredictionResponse } from '../types/api';

interface PredictionState {
  activePredictions: Map<string, PredictionResponse>;
  addPrediction: (prediction: PredictionResponse) => void;
  updatePrediction: (id: string, updates: Partial<PredictionResponse>) => void;
  removePrediction: (id: string) => void;
  clearPredictions: () => void;
}

export const usePredictionStore = create<PredictionState>((set) => ({
  activePredictions: new Map(),
  
  addPrediction: (prediction) =>
    set((state) => {
      const newPredictions = new Map(state.activePredictions);
      newPredictions.set(prediction.id, prediction);
      return { activePredictions: newPredictions };
    }),
  
  updatePrediction: (id, updates) =>
    set((state) => {
      const newPredictions = new Map(state.activePredictions);
      const existing = newPredictions.get(id);
      if (existing) {
        newPredictions.set(id, { ...existing, ...updates });
      }
      return { activePredictions: newPredictions };
    }),
  
  removePrediction: (id) =>
    set((state) => {
      const newPredictions = new Map(state.activePredictions);
      newPredictions.delete(id);
      return { activePredictions: newPredictions };
    }),
  
  clearPredictions: () => set({ activePredictions: new Map() }),
}));
