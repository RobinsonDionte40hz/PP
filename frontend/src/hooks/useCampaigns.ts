import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { campaignService } from '../services/campaignService';
import type { CampaignCreate } from '../types/api';

// Query keys
export const campaignKeys = {
  all: ['campaigns'] as const,
  lists: () => [...campaignKeys.all, 'list'] as const,
  list: (filters?: Record<string, unknown>) => [...campaignKeys.lists(), filters] as const,
  details: () => [...campaignKeys.all, 'detail'] as const,
  detail: (id: string) => [...campaignKeys.details(), id] as const,
  statistics: (id: string) => [...campaignKeys.detail(id), 'statistics'] as const,
  phase: (id: string, phaseNum: number) => [...campaignKeys.detail(id), 'phase', phaseNum] as const,
};

// Get single campaign
export function useCampaign(id: string) {
  return useQuery({
    queryKey: campaignKeys.detail(id),
    queryFn: () => campaignService.getCampaign(id),
    refetchInterval: (query) => {
      // Keep refetching while campaign is running
      const data = query.state.data;
      if (data?.status === 'running' || data?.status === 'pending') {
        return 10000; // 10 seconds
      }
      return false;
    },
  });
}

// List campaigns
export function useCampaigns(filters?: { status?: string; limit?: number; offset?: number }) {
  return useQuery({
    queryKey: campaignKeys.list(filters),
    queryFn: () => campaignService.listCampaigns(filters),
  });
}

// Get campaign statistics
export function useCampaignStatistics(id: string) {
  return useQuery({
    queryKey: campaignKeys.statistics(id),
    queryFn: () => campaignService.getCampaignStatistics(id),
  });
}

// Get phase details
export function usePhaseDetails(id: string, phaseNum: number) {
  return useQuery({
    queryKey: campaignKeys.phase(id, phaseNum),
    queryFn: () => campaignService.getPhaseDetails(id, phaseNum),
  });
}

// Create campaign mutation
export function useCreateCampaign() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: CampaignCreate) => campaignService.createCampaign(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: campaignKeys.lists() });
    },
  });
}

// Resume campaign mutation
export function useResumeCampaign() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => campaignService.resumeCampaign(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: campaignKeys.detail(id) });
    },
  });
}

// Delete campaign mutation
export function useDeleteCampaign() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => campaignService.deleteCampaign(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: campaignKeys.lists() });
    },
  });
}
