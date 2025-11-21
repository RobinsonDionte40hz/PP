import React from 'react';
import { Card, CardContent, Skeleton, Box } from '@mui/material';

export const MetricCardSkeleton: React.FC = () => {
  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
          <Skeleton variant="text" width="60%" height={24} />
          <Skeleton variant="circular" width={24} height={24} />
        </Box>
        <Skeleton variant="text" width="80%" height={48} sx={{ my: 1 }} />
        <Skeleton variant="text" width="40%" height={20} />
      </CardContent>
    </Card>
  );
};

export const MetricsGridSkeleton: React.FC<{ count?: number }> = ({ count = 6 }) => {
  return (
    <>
      {Array.from({ length: count }).map((_, i) => (
        <MetricCardSkeleton key={i} />
      ))}
    </>
  );
};
