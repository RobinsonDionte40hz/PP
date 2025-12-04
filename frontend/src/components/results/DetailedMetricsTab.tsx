import React, { useState } from 'react';
import {
  Box,
  Stack,
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  ToggleButton,
  ToggleButtonGroup,
  Divider,
} from '@mui/material';
import {
  ShowChart,
  TableChart,
} from '@mui/icons-material';
import type { PredictionResponse } from '../../types/api';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from 'recharts';

interface DetailedMetricsTabProps {
  prediction: PredictionResponse;
}

const DetailedMetricsTab: React.FC<DetailedMetricsTabProps> = ({ prediction }) => {
  const [viewMode, setViewMode] = useState<'charts' | 'table'>('charts');

  // Extract metrics from nested object
  const metrics = prediction.metrics || {};
  const bestRMSD = metrics.best_rmsd ?? metrics.final_rmsd ?? null;
  const finalRMSD = metrics.final_rmsd ?? metrics.best_rmsd ?? null;
  const bestEnergy = metrics.best_energy ?? metrics.final_energy ?? null;
  const finalEnergy = metrics.final_energy ?? metrics.best_energy ?? null;
  const energyChange = metrics.energy_change ?? null;
  const convergenceRate = metrics.convergence_rate ?? null;
  const finalAggressiveness = metrics.final_aggressiveness ?? null;
  const finalConsistency = metrics.final_consistency ?? null;
  const conformationsExplored = metrics.conformations_explored ?? prediction.current_iteration ?? null;
  const uniqueStructures = metrics.unique_structures ?? null;
  const gdtTS = metrics.gdt_ts_score ?? null;
  const tmScore = metrics.tm_score ?? null;
  const qaapAlignment = metrics.qaap_alignment ?? null;
  const resonance40hz = metrics.resonance_40hz ?? null;
  const waterShielding = metrics.water_shielding ?? null;
  const qcpScore = metrics.qcp_score ?? null;

  // Generate mock convergence data (in production, fetch from backend)
  const generateConvergenceData = () => {
    const iterations = prediction.total_iterations || 5000;
    const points = Math.min(100, iterations);
    const step = Math.floor(iterations / points);

    return Array.from({ length: points }, (_, i) => {
      const iter = i * step;
      const progress = iter / iterations;

      return {
        iteration: iter,
        energy: -50 - progress * 150 + Math.random() * 30,
        rmsd: 15 - progress * 10 + Math.random() * 2,
        aggressiveness: 9 + Math.sin(progress * Math.PI * 4) * 3,
        consistency: 0.4 + progress * 0.4 + Math.random() * 0.1,
      };
    });
  };

  const convergenceData = generateConvergenceData();

  // Quality metrics with actual data
  const qualityMetrics = [
    {
      category: 'Structure Quality',
      metrics: [
        { name: 'Best RMSD', value: bestRMSD !== null && bestRMSD !== Infinity ? bestRMSD.toFixed(3) : 'N/A', unit: bestRMSD !== Infinity ? 'Å' : '', threshold: '< 5.0', status: bestRMSD !== null && bestRMSD !== Infinity && bestRMSD < 5 ? 'pass' : bestRMSD === Infinity ? 'unknown' : 'fail' },
        { name: 'Final RMSD', value: finalRMSD !== null && finalRMSD !== Infinity ? finalRMSD.toFixed(3) : 'N/A', unit: finalRMSD !== Infinity ? 'Å' : '', threshold: '< 5.0', status: finalRMSD !== null && finalRMSD !== Infinity && finalRMSD < 5 ? 'pass' : finalRMSD === Infinity ? 'unknown' : 'fail' },
        { name: 'GDT-TS', value: gdtTS !== null ? gdtTS.toFixed(1) : 'N/A', unit: '', threshold: '> 50', status: gdtTS !== null && gdtTS > 50 ? 'pass' : gdtTS !== null ? 'fail' : 'unknown' },
        { name: 'TM-Score', value: tmScore !== null ? tmScore.toFixed(3) : 'N/A', unit: '', threshold: '> 0.5', status: tmScore !== null && tmScore > 0.5 ? 'pass' : tmScore !== null ? 'fail' : 'unknown' },
      ],
    },
    {
      category: 'Energy Metrics',
      metrics: [
        { name: 'Best Energy', value: bestEnergy !== null ? bestEnergy.toFixed(2) : 'N/A', unit: 'kcal/mol', threshold: '< 0', status: bestEnergy !== null && bestEnergy < 0 ? 'pass' : bestEnergy !== null ? 'fail' : 'unknown' },
        { name: 'Final Energy', value: finalEnergy !== null ? finalEnergy.toFixed(2) : 'N/A', unit: 'kcal/mol', threshold: '< 0', status: finalEnergy !== null && finalEnergy < 0 ? 'pass' : finalEnergy !== null ? 'fail' : 'unknown' },
        { name: 'Energy Change', value: energyChange !== null ? energyChange.toFixed(2) : 'N/A', unit: 'kcal/mol', threshold: 'Decreasing', status: energyChange !== null && energyChange < 0 ? 'pass' : energyChange !== null ? 'warning' : 'unknown' },
        { name: 'Convergence Rate', value: convergenceRate !== null ? convergenceRate.toFixed(1) : 'N/A', unit: '%', threshold: '> 80', status: convergenceRate !== null && convergenceRate > 80 ? 'pass' : convergenceRate !== null ? 'warning' : 'unknown' },
      ],
    },
    {
      category: 'Exploration Metrics',
      metrics: [
        { name: 'Final Aggressiveness', value: finalAggressiveness !== null ? finalAggressiveness.toFixed(2) : 'N/A', unit: '', threshold: '3.0 - 15.0', status: finalAggressiveness !== null && finalAggressiveness >= 3.0 && finalAggressiveness <= 15.0 ? 'pass' : finalAggressiveness !== null ? 'warning' : 'unknown' },
        { name: 'Final Consistency', value: finalConsistency !== null ? finalConsistency.toFixed(3) : 'N/A', unit: '', threshold: '0.2 - 1.0', status: finalConsistency !== null && finalConsistency >= 0.2 && finalConsistency <= 1.0 ? 'pass' : finalConsistency !== null ? 'warning' : 'unknown' },
        { name: 'Conformations Explored', value: conformationsExplored !== null ? conformationsExplored.toLocaleString() : 'N/A', unit: '', threshold: 'N/A', status: conformationsExplored !== null ? 'pass' : 'unknown' },
        { name: 'Unique Structures', value: uniqueStructures !== null ? uniqueStructures.toLocaleString() : 'N/A', unit: '', threshold: 'N/A', status: uniqueStructures !== null ? 'pass' : 'unknown' },
      ],
    },
    {
      category: 'Quantum Metrics',
      metrics: [
        { name: 'QAAP Alignment', value: qaapAlignment !== null ? qaapAlignment.toFixed(3) : 'N/A', unit: '', threshold: '0.7 - 1.3', status: qaapAlignment !== null && qaapAlignment >= 0.7 && qaapAlignment <= 1.3 ? 'pass' : qaapAlignment !== null ? 'warning' : 'unknown' },
        { name: '40 Hz Resonance', value: resonance40hz !== null ? resonance40hz.toFixed(3) : 'N/A', unit: '', threshold: '0.9 - 1.2', status: resonance40hz !== null && resonance40hz >= 0.9 && resonance40hz <= 1.2 ? 'pass' : resonance40hz !== null ? 'warning' : 'unknown' },
        { name: 'Water Shielding', value: waterShielding !== null ? waterShielding.toFixed(1) : 'N/A', unit: 'fs', threshold: '~ 408', status: waterShielding !== null && Math.abs(waterShielding - 408) < 50 ? 'pass' : waterShielding !== null ? 'warning' : 'unknown' },
        { name: 'QCP Score', value: qcpScore !== null ? qcpScore.toFixed(2) : 'N/A', unit: '', threshold: '> 4.0', status: qcpScore !== null && qcpScore > 4.0 ? 'pass' : qcpScore !== null ? 'warning' : 'unknown' },
      ],
    },
  ];

  // Memory statistics (mock data)
  const memoryStats = [
    { metric: 'Total Memories', value: '1,247', description: 'Stored experiences' },
    { metric: 'Shared Memories', value: '342', description: 'Significance ≥ 0.7' },
    { metric: 'High-Impact Memories', value: '89', description: 'Significance ≥ 0.9' },
    { metric: 'Memory Retrievals', value: '5,623', description: 'Query operations' },
    { metric: 'Avg Retrieval Time', value: '4.2 μs', description: 'Query performance' },
    { metric: 'Cache Hit Rate', value: '87.3%', description: 'Memory efficiency' },
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pass': return 'success';
      case 'fail': return 'error';
      case 'warning': return 'warning';
      default: return 'default';
    }
  };

  return (
    <Stack spacing={3}>
      {/* View Mode Toggle */}
      <Box display="flex" justifyContent="flex-end">
        <ToggleButtonGroup
          value={viewMode}
          exclusive
          onChange={(_, newMode) => newMode && setViewMode(newMode)}
          size="small"
        >
          <ToggleButton value="charts">
            <ShowChart sx={{ mr: 1 }} />
            Charts
          </ToggleButton>
          <ToggleButton value="table">
            <TableChart sx={{ mr: 1 }} />
            Table
          </ToggleButton>
        </ToggleButtonGroup>
      </Box>

      {viewMode === 'charts' ? (
        <>
          {/* Energy Convergence */}
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Energy Convergence
            </Typography>
            <Typography variant="caption" color="text.secondary" display="block" mb={2}>
              System energy over optimization iterations
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={convergenceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="iteration" label={{ value: 'Iteration', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Energy (kcal/mol)', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <ReferenceLine y={0} stroke="#666" strokeDasharray="3 3" label="Stability Threshold" />
                <Area
                  type="monotone"
                  dataKey="energy"
                  stroke="#8884d8"
                  fill="#8884d8"
                  fillOpacity={0.6}
                  name="Energy"
                />
              </AreaChart>
            </ResponsiveContainer>
          </Paper>

          {/* RMSD Convergence */}
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              RMSD Convergence
            </Typography>
            <Typography variant="caption" color="text.secondary" display="block" mb={2}>
              Structural deviation from native state
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={convergenceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="iteration" label={{ value: 'Iteration', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'RMSD (Å)', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <ReferenceLine y={5} stroke="#ff9800" strokeDasharray="3 3" label="Acceptable" />
                <ReferenceLine y={2} stroke="#4caf50" strokeDasharray="3 3" label="Excellent" />
                <Line
                  type="monotone"
                  dataKey="rmsd"
                  stroke="#82ca9d"
                  strokeWidth={2}
                  dot={false}
                  name="RMSD"
                />
              </LineChart>
            </ResponsiveContainer>
          </Paper>

          {/* Exploration Parameters */}
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Exploration Parameters Evolution
            </Typography>
            <Typography variant="caption" color="text.secondary" display="block" mb={2}>
              Agent behavior adaptation over time
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={convergenceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="iteration" label={{ value: 'Iteration', position: 'insideBottom', offset: -5 }} />
                <YAxis yAxisId="left" label={{ value: 'Aggressiveness', angle: -90, position: 'insideLeft' }} />
                <YAxis yAxisId="right" orientation="right" label={{ value: 'Consistency', angle: 90, position: 'insideRight' }} />
                <Tooltip />
                <Legend />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="aggressiveness"
                  stroke="#ff9800"
                  strokeWidth={2}
                  dot={false}
                  name="Aggressiveness"
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="consistency"
                  stroke="#9c27b0"
                  strokeWidth={2}
                  dot={false}
                  name="Consistency"
                />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </>
      ) : (
        <>
          {/* Quality Metrics Tables */}
          {qualityMetrics.map((category, idx) => (
            <Paper key={idx} sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                {category.category}
              </Typography>
              <Divider sx={{ my: 2 }} />
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell><strong>Metric</strong></TableCell>
                      <TableCell align="right"><strong>Value</strong></TableCell>
                      <TableCell align="right"><strong>Threshold</strong></TableCell>
                      <TableCell align="center"><strong>Status</strong></TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {category.metrics.map((metric, metricIdx) => (
                      <TableRow key={metricIdx}>
                        <TableCell>{metric.name}</TableCell>
                        <TableCell align="right">
                          {metric.value} {metric.unit}
                        </TableCell>
                        <TableCell align="right">{metric.threshold}</TableCell>
                        <TableCell align="center">
                          <Chip
                            label={metric.status.toUpperCase()}
                            color={getStatusColor(metric.status)}
                            size="small"
                          />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>
          ))}

          {/* Memory Statistics */}
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Memory System Statistics
            </Typography>
            <Divider sx={{ my: 2 }} />
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell><strong>Metric</strong></TableCell>
                    <TableCell align="right"><strong>Value</strong></TableCell>
                    <TableCell><strong>Description</strong></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {memoryStats.map((stat, idx) => (
                    <TableRow key={idx}>
                      <TableCell>{stat.metric}</TableCell>
                      <TableCell align="right">
                        <Typography variant="body2" fontWeight="bold">
                          {stat.value}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="caption" color="text.secondary">
                          {stat.description}
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </>
      )}
    </Stack>
  );
};

export default DetailedMetricsTab;
