import React from 'react';
import { Box, Card, CardContent, Tabs, Tab, Skeleton, Grid } from '@mui/material';

export const ResultsAnalysisSkeleton: React.FC = () => {
  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Skeleton variant="text" width={300} height={40} />
        <Skeleton variant="text" width={200} height={24} sx={{ mt: 1 }} />
      </Box>

      {/* Tabs */}
      <Card sx={{ mb: 3 }}>
        <Tabs value={0}>
          {['Summary', 'Metrics', 'Trajectory', 'Geometric'].map((tab, i) => (
            <Tab key={i} label={<Skeleton variant="text" width={80} />} />
          ))}
        </Tabs>
      </Card>

      {/* Content Grid */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Skeleton variant="text" width={150} height={32} sx={{ mb: 2 }} />
              {[1, 2, 3, 4].map((i) => (
                <Box key={i} sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                  <Skeleton variant="text" width="40%" />
                  <Skeleton variant="text" width="30%" />
                </Box>
              ))}
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Skeleton variant="text" width={150} height={32} sx={{ mb: 2 }} />
              <Skeleton variant="rectangular" height={250} />
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};
