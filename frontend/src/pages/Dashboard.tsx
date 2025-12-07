import React, { useEffect } from 'react';
import {
  Box,
  Stack,
  Typography,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { predictionService } from '../services';
import {
  QuickActionsCard,
  SystemStatusCard,
  RecentPredictionsCard,
  StatisticsCard,
  FeedbackCard,
} from '../components/dashboard';
import { DashboardSkeleton } from '../components/common/skeletons';
import { animationStyles } from '../utils/animations';
import QuotaDisplay from '../components/common/QuotaDisplay';
import EmailVerificationBanner from '../components/common/EmailVerificationBanner';

const Dashboard: React.FC = () => {
  const navigate = useNavigate();

  // Fetch recent predictions
  const { data: predictions, isLoading: predictionsLoading, refetch } = useQuery({
    queryKey: ['predictions', { limit: 10, status: 'recent' }],
    queryFn: () => predictionService.listPredictions({ limit: 10 }),
    refetchInterval: 30000, // Auto-refresh every 30 seconds
  });

  // Fetch statistics
  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['dashboard-stats'],
    queryFn: async () => {
      // Get all predictions for statistics (use max allowed page_size of 100)
      const allPredictions = await predictionService.listPredictions({ limit: 100 });
      
      const total = allPredictions.length;
      const running = allPredictions.filter(p => p.status === 'running').length;
      const completed = allPredictions.filter(p => p.status === 'completed').length;
      const failed = allPredictions.filter(p => p.status === 'failed').length;
      
      // Calculate average RMSD for completed predictions
      // Use best_rmsd (native comparison) if available, otherwise fall back to folding_rmsd
      const completedWithRMSD = allPredictions.filter(
        p => p.status === 'completed' && (
          (p.best_rmsd !== null && p.best_rmsd !== undefined && p.best_rmsd !== Infinity) ||
          (p.metrics?.folding_rmsd !== null && p.metrics?.folding_rmsd !== undefined)
        )
      );
      const avgRMSD = completedWithRMSD.length > 0
        ? completedWithRMSD.reduce((sum, p) => {
            // Prefer best_rmsd (native comparison), fall back to folding_rmsd
            const rmsdValue = (p.best_rmsd !== null && p.best_rmsd !== undefined && p.best_rmsd !== Infinity)
              ? p.best_rmsd
              : (p.metrics?.folding_rmsd || 0);
            return sum + rmsdValue;
          }, 0) / completedWithRMSD.length
        : 0;

      // Calculate success rate (completed vs failed) - handle division by zero
      const completedAndFailed = completed + failed;
      const successRate = completedAndFailed > 0 ? (completed / completedAndFailed) * 100 : 100;

      return {
        total,
        running,
        completed,
        failed,
        avgRMSD,
        successRate,
      };
    },
    refetchInterval: 30000,
  });

  useEffect(() => {
    // Set up auto-refresh
    const interval = setInterval(() => {
      refetch();
    }, 30000);

    return () => clearInterval(interval);
  }, [refetch]);

  if (predictionsLoading || statsLoading) {
    return (
      <Box sx={{ p: 3 }}>
        <Box mb={4}>
          <Typography variant="h4" fontWeight="bold" gutterBottom>
            Dashboard
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Welcome to the Dual-System Protein Platform
          </Typography>
        </Box>
        <DashboardSkeleton />
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3, ...animationStyles.fadeIn }}>
      {/* Email Verification Banner */}
      <EmailVerificationBanner />

      {/* Header */}
      <Box mb={4} sx={animationStyles.fadeInDown}>
        <Typography variant="h4" fontWeight="bold" gutterBottom>
          Dashboard
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Welcome to the Dual-System Protein Platform
        </Typography>
      </Box>

      {/* Main Grid */}
      <Stack spacing={3}>
        {/* Quota Display & Quick Actions & System Status Row */}
        <Box display="flex" flexDirection={{ xs: 'column', md: 'row' }} gap={3}>
          <Box flex={1}>
            <QuotaDisplay variant="full" />
          </Box>
          <Box flex={1}>
            <QuickActionsCard onNavigate={navigate} />
          </Box>
          <Box flex={1}>
            <SystemStatusCard stats={stats} />
          </Box>
        </Box>

        {/* Statistics */}
        <StatisticsCard stats={stats} predictions={predictions || []} />

        {/* Recent Predictions */}
        <RecentPredictionsCard
          predictions={predictions || []}
          onNavigate={navigate}
          onRefresh={refetch}
        />

        {/* Feedback Section */}
        <FeedbackCard />
      </Stack>
    </Box>
  );
};

export default Dashboard;
