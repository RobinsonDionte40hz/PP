import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { predictionService } from '../services/predictionService';
import type { PredictionCreate } from '../types/api';

// Query keys
export const predictionKeys = {
  all: ['predictions'] as const,
  lists: () => [...predictionKeys.all, 'list'] as const,
  list: (filters?: Record<string, unknown>) => [...predictionKeys.lists(), filters] as const,
  details: () => [...predictionKeys.all, 'detail'] as const,
  detail: (id: string) => [...predictionKeys.details(), id] as const,
};

// Get single prediction
export function usePrediction(id: string) {
  return useQuery({
    queryKey: predictionKeys.detail(id),
    queryFn: () => predictionService.getPrediction(id),
    refetchInterval: (query) => {
      // Keep refetching while prediction is running or queued
      const data = query.state.data;
      if (data?.status === 'running' || data?.status === 'pending' || data?.status === 'queued') {
        return 5000; // 5 seconds
      }
      return false;
    },
  });
}

// List predictions
export function usePredictions(filters?: { status?: string; limit?: number; offset?: number }) {
  return useQuery({
    queryKey: predictionKeys.list(filters),
    queryFn: () => predictionService.listPredictions(filters),
  });
}

// Create prediction mutation
export function useCreatePrediction() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: PredictionCreate) => predictionService.createPrediction(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: predictionKeys.lists() });
    },
  });
}

// Pause prediction mutation
export function usePausePrediction() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => predictionService.pausePrediction(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: predictionKeys.detail(id) });
    },
  });
}

// Resume prediction mutation
export function useResumePrediction() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => predictionService.resumePrediction(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: predictionKeys.detail(id) });
    },
  });
}

// Stop prediction mutation
export function useStopPrediction() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => predictionService.stopPrediction(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: predictionKeys.detail(id) });
    },
  });
}

// Delete prediction mutation
export function useDeletePrediction() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => predictionService.deletePrediction(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: predictionKeys.lists() });
    },
  });
}

// Download checkpoint mutation
export function useDownloadCheckpoint() {
  return useMutation({
    mutationFn: async (id: string) => {
      const blob = await predictionService.downloadCheckpoint(id);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `checkpoint_${id}.json`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    },
  });
}
