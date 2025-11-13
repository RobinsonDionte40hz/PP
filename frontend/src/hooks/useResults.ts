import { useQuery, useMutation } from '@tanstack/react-query';
import { resultService } from '../services/resultService';

// Query keys
export const resultKeys = {
  all: ['results'] as const,
  details: () => [...resultKeys.all, 'detail'] as const,
  detail: (id: string) => [...resultKeys.details(), id] as const,
  structure: (id: string) => [...resultKeys.detail(id), 'structure'] as const,
  trajectory: (id: string) => [...resultKeys.detail(id), 'trajectory'] as const,
  metrics: (id: string) => [...resultKeys.detail(id), 'metrics'] as const,
};

// Get result detail
export function useResultDetail(id: string) {
  return useQuery({
    queryKey: resultKeys.detail(id),
    queryFn: () => resultService.getResultDetail(id),
  });
}

// Get structure data
export function useStructure(id: string) {
  return useQuery({
    queryKey: resultKeys.structure(id),
    queryFn: () => resultService.getStructure(id),
  });
}

// Get trajectory data
export function useTrajectory(id: string) {
  return useQuery({
    queryKey: resultKeys.trajectory(id),
    queryFn: () => resultService.getTrajectory(id),
  });
}

// Get metrics
export function useMetrics(id: string) {
  return useQuery({
    queryKey: resultKeys.metrics(id),
    queryFn: () => resultService.getMetrics(id),
  });
}

// Export results mutation
export function useExportResults() {
  return useMutation({
    mutationFn: async ({ id, format }: { id: string; format: 'json' | 'pdf' }) => {
      const blob = await resultService.exportResults(id, format);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `results_${id}.${format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    },
  });
}

// Compare results mutation
export function useCompareResults() {
  return useMutation({
    mutationFn: (ids: string[]) => resultService.compareResults(ids),
  });
}
