import React from 'react';
import { Box, Card, CardContent, Skeleton } from '@mui/material';

export const DashboardSkeleton: React.FC = () => {
  return (
    <Box>
      {/* Quick Actions Card */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Skeleton variant="text" width={150} height={32} sx={{ mb: 2 }} />
          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 2 }}>
            {[1, 2, 3, 4].map((i) => (
              <Skeleton key={i} variant="rectangular" height={120} sx={{ borderRadius: 1 }} />
            ))}
          </Box>
        </CardContent>
      </Card>

      {/* System Status and Recent Predictions */}
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 3 }}>
        <Card>
          <CardContent>
            <Skeleton variant="text" width={120} height={32} sx={{ mb: 2 }} />
            {[1, 2, 3, 4].map((i) => (
              <Box key={i} sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Skeleton variant="circular" width={40} height={40} sx={{ mr: 2 }} />
                <Box sx={{ flex: 1 }}>
                  <Skeleton variant="text" width="60%" />
                  <Skeleton variant="text" width="40%" />
                </Box>
              </Box>
            ))}
          </CardContent>
        </Card>
        <Card>
          <CardContent>
            <Skeleton variant="text" width={150} height={32} sx={{ mb: 2 }} />
            {[1, 2, 3].map((i) => (
              <Box key={i} sx={{ mb: 2, pb: 2, borderBottom: i < 3 ? '1px solid #e0e0e0' : 'none' }}>
                <Skeleton variant="text" width="80%" />
                <Skeleton variant="text" width="60%" />
                <Skeleton variant="text" width="40%" />
              </Box>
            ))}
          </CardContent>
        </Card>
      </Box>

      {/* Statistics Cards */}
      <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 3, mt: 3 }}>
        {[1, 2, 3, 4].map((i) => (
          <Card key={i}>
            <CardContent>
              <Skeleton variant="text" width="60%" />
              <Skeleton variant="text" width="80%" height={48} sx={{ my: 1 }} />
              <Skeleton variant="text" width="40%" />
            </CardContent>
          </Card>
        ))}
      </Box>
    </Box>
  );
};
