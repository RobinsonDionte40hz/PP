import React from 'react';
import { Card, CardContent, Skeleton, Box, Divider } from '@mui/material';

export const ChartSkeleton: React.FC = () => {
  return (
    <Card>
      <CardContent>
        <Skeleton variant="text" width={200} height={32} sx={{ mb: 2 }} />
        <Box sx={{ height: 300, display: 'flex', alignItems: 'flex-end', gap: 1 }}>
          {Array.from({ length: 12 }).map((_, i) => (
            <Skeleton
              key={i}
              variant="rectangular"
              width="100%"
              height={Math.random() * 250 + 50}
              sx={{ borderRadius: '4px 4px 0 0' }}
            />
          ))}
        </Box>
      </CardContent>
    </Card>
  );
};

export const LiveChartsSkeleton: React.FC = () => {
  return (
    <Box>
      <Skeleton variant="text" width={150} height={32} sx={{ mb: 2 }} />
      <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: 3 }}>
        {[1, 2, 3].map((i) => (
          <ChartSkeleton key={i} />
        ))}
      </Box>
    </Box>
  );
};
