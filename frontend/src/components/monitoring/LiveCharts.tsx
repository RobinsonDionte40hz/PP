import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  ToggleButtonGroup,
  ToggleButton,
  useTheme,
  alpha,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import type { PredictionProgress } from '../../types/api';

interface LiveChartsProps {
  progressData: PredictionProgress[];
}

type ChartType = 'energy' | 'rmsd' | 'params' | 'all';

const LiveCharts: React.FC<LiveChartsProps> = ({ progressData }) => {
  const theme = useTheme();
  const [chartType, setChartType] = useState<ChartType>('energy');

  const renderEnergyChart = () => (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={progressData}>
        <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
        <XAxis
          dataKey="iteration"
          tick={{ fontSize: 12 }}
          stroke={theme.palette.text.secondary}
          label={{ value: 'Iteration', position: 'insideBottom', offset: -5 }}
        />
        <YAxis
          tick={{ fontSize: 12 }}
          stroke={theme.palette.text.secondary}
          label={{ value: 'Energy (kcal/mol)', angle: -90, position: 'insideLeft' }}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: theme.palette.background.paper,
            border: `1px solid ${theme.palette.divider}`,
            borderRadius: 4,
          }}
        />
        <Legend />
        <Line
          type="monotone"
          dataKey="energy"
          stroke={theme.palette.primary.main}
          strokeWidth={2}
          dot={false}
          name="Current Energy"
        />
        <Line
          type="monotone"
          dataKey="best_energy"
          stroke={theme.palette.success.main}
          strokeWidth={2}
          dot={false}
          name="Best Energy"
        />
      </LineChart>
    </ResponsiveContainer>
  );

  const renderRMSDChart = () => (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={progressData}>
        <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
        <XAxis
          dataKey="iteration"
          tick={{ fontSize: 12 }}
          stroke={theme.palette.text.secondary}
          label={{ value: 'Iteration', position: 'insideBottom', offset: -5 }}
        />
        <YAxis
          tick={{ fontSize: 12 }}
          stroke={theme.palette.text.secondary}
          label={{ value: 'RMSD (Å)', angle: -90, position: 'insideLeft' }}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: theme.palette.background.paper,
            border: `1px solid ${theme.palette.divider}`,
            borderRadius: 4,
          }}
        />
        <Legend />
        <Line
          type="monotone"
          dataKey="rmsd"
          stroke={theme.palette.info.main}
          strokeWidth={2}
          dot={false}
          name="Current RMSD"
        />
        <Line
          type="monotone"
          dataKey="best_rmsd"
          stroke={theme.palette.success.main}
          strokeWidth={2}
          dot={false}
          name="Best RMSD"
        />
      </LineChart>
    </ResponsiveContainer>
  );

  const renderParamsChart = () => (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={progressData}>
        <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
        <XAxis
          dataKey="iteration"
          tick={{ fontSize: 12 }}
          stroke={theme.palette.text.secondary}
          label={{ value: 'Iteration', position: 'insideBottom', offset: -5 }}
        />
        <YAxis
          tick={{ fontSize: 12 }}
          stroke={theme.palette.text.secondary}
          label={{ value: 'Value', angle: -90, position: 'insideLeft' }}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: theme.palette.background.paper,
            border: `1px solid ${theme.palette.divider}`,
            borderRadius: 4,
          }}
        />
        <Legend />
        <Line
          type="monotone"
          dataKey="aggressiveness"
          stroke={theme.palette.warning.main}
          strokeWidth={2}
          dot={false}
          name="Aggressiveness"
        />
        <Line
          type="monotone"
          dataKey="consistency"
          stroke={theme.palette.secondary.main}
          strokeWidth={2}
          dot={false}
          name="Consistency"
        />
      </LineChart>
    </ResponsiveContainer>
  );

  const renderChart = () => {
    if (progressData.length === 0) {
      return (
        <Box
          display="flex"
          justifyContent="center"
          alignItems="center"
          height={300}
          sx={{
            backgroundColor: alpha(theme.palette.background.default, 0.5),
            borderRadius: 1,
          }}
        >
          <Typography variant="body2" color="text.secondary">
            Waiting for data...
          </Typography>
        </Box>
      );
    }

    switch (chartType) {
      case 'energy':
        return renderEnergyChart();
      case 'rmsd':
        return renderRMSDChart();
      case 'params':
        return renderParamsChart();
      case 'all':
        return (
          <Box>
            {renderEnergyChart()}
            <Box mt={3}>{renderRMSDChart()}</Box>
            <Box mt={3}>{renderParamsChart()}</Box>
          </Box>
        );
      default:
        return renderEnergyChart();
    }
  };

  return (
    <Paper elevation={2} sx={{ p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h6" fontWeight="bold">
          Live Charts
        </Typography>
        <ToggleButtonGroup
          value={chartType}
          exclusive
          onChange={(_, newValue) => newValue && setChartType(newValue)}
          size="small"
        >
          <ToggleButton value="energy">Energy</ToggleButton>
          <ToggleButton value="rmsd">RMSD</ToggleButton>
          <ToggleButton value="params">Parameters</ToggleButton>
          <ToggleButton value="all">All</ToggleButton>
        </ToggleButtonGroup>
      </Box>

      {renderChart()}

      <Box mt={2} display="flex" justifyContent="space-between">
        <Typography variant="caption" color="text.secondary">
          Data points: {progressData.length}
        </Typography>
        {progressData.length > 0 && (
          <Typography variant="caption" color="text.secondary">
            Latest update: {new Date(progressData[progressData.length - 1].timestamp).toLocaleTimeString()}
          </Typography>
        )}
      </Box>
    </Paper>
  );
};

export default LiveCharts;
