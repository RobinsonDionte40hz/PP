import React from 'react';
import {
  Box,
  Stack,
  Paper,
  Typography,
} from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';

interface CampaignStatisticsProps {
  statistics: {
    total_proteins?: number;
    successful_predictions?: number;
    failed_predictions?: number;
    average_rmsd?: number;
    average_energy?: number;
    average_iterations?: number;
    quality_distribution?: { [key: string]: number };
  };
}

const COLORS = {
  excellent: '#4caf50',
  good: '#2196f3',
  acceptable: '#ff9800',
  poor: '#f44336',
};

const CampaignStatistics: React.FC<CampaignStatisticsProps> = ({ statistics }) => {
  // Prepare data for quality distribution pie chart
  const qualityData = statistics.quality_distribution
    ? Object.entries(statistics.quality_distribution).map(([quality, count]) => ({
        name: quality.charAt(0).toUpperCase() + quality.slice(1),
        value: count,
      }))
    : [];

  // Prepare data for success/failure bar chart
  const successData = [
    {
      name: 'Results',
      Successful: statistics.successful_predictions || 0,
      Failed: statistics.failed_predictions || 0,
    },
  ];

  return (
    <Stack spacing={3}>
      {/* Summary Statistics */}
      <Stack
        direction={{ xs: 'column', md: 'row' }}
        spacing={3}
      >
        <Box sx={{ flex: 1 }}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Typography variant="subtitle1" gutterBottom fontWeight="medium">
              Summary Statistics
            </Typography>
            <Box sx={{ mt: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Total Proteins:
                </Typography>
                <Typography variant="body2" fontWeight="medium">
                  {statistics.total_proteins || 0}
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Success Rate:
                </Typography>
                <Typography variant="body2" fontWeight="medium" color="success.main">
                  {statistics.total_proteins
                    ? `${(
                        ((statistics.successful_predictions || 0) /
                          statistics.total_proteins) *
                        100
                      ).toFixed(1)}%`
                    : 'N/A'}
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Average RMSD:
                </Typography>
                <Typography variant="body2" fontWeight="medium">
                  {statistics.average_rmsd
                    ? `${statistics.average_rmsd.toFixed(2)} Å`
                    : 'N/A'}
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Average Energy:
                </Typography>
                <Typography variant="body2" fontWeight="medium">
                  {statistics.average_energy
                    ? `${statistics.average_energy.toFixed(2)} kcal/mol`
                    : 'N/A'}
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="body2" color="text.secondary">
                  Avg Iterations:
                </Typography>
                <Typography variant="body2" fontWeight="medium">
                  {statistics.average_iterations
                    ? Math.round(statistics.average_iterations)
                    : 'N/A'}
                </Typography>
              </Box>
            </Box>
          </Paper>
        </Box>

        {/* Success/Failure Chart */}
        <Box sx={{ flex: 1 }}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Typography variant="subtitle1" gutterBottom fontWeight="medium">
              Success vs Failure
            </Typography>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={successData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="Successful" fill="#4caf50" />
                <Bar dataKey="Failed" fill="#f44336" />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Box>
      </Stack>

      {/* Quality Distribution */}
      {qualityData.length > 0 && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="subtitle1" gutterBottom fontWeight="medium">
            Quality Distribution
          </Typography>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={qualityData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) =>
                  `${name}: ${percent ? (percent * 100).toFixed(0) : 0}%`
                }
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {qualityData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={COLORS[entry.name.toLowerCase() as keyof typeof COLORS] || '#999'}
                  />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Paper>
      )}
    </Stack>
  );
};

export default CampaignStatistics;
