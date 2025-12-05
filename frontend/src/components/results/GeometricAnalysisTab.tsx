import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Box,
  Stack,
  Paper,
  Typography,
  ToggleButton,
  ToggleButtonGroup,
  Chip,
  Alert,
  CircularProgress,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Divider,
} from '@mui/material';
import {
  Hexagon,
  ChangeHistory,
  Crop169,
} from '@mui/icons-material';
import {
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
} from 'recharts';

interface GeometricAnalysisTabProps {
  predictionId: string;
}

interface GeometricPattern {
  name: string;
  score: number;
  count: number;
  percentage: number;
  icon: React.ReactNode;
  color: string;
}



const GeometricAnalysisTab: React.FC<GeometricAnalysisTabProps> = ({ predictionId }) => {
  const [viewMode, setViewMode] = useState<'overview' | 'details'>('overview');

  // Fetch geometric analysis data from backend
  const { data: geometricData, isLoading, error } = useQuery({
    queryKey: ['geometric-analysis', predictionId],
    queryFn: async () => {
      const response = await fetch(`/api/results/${predictionId}/geometric`);
      if (!response.ok) {
        throw new Error('Failed to fetch geometric analysis');
      }
      const data = await response.json();
      
      // If no geometric analysis available, return null
      if (!data.geometric_analysis) {
        return null;
      }
      
      const geo = data.geometric_analysis;
      const qcpp = data.qcpp_metrics || {};
      
      // Extract platonic similarities (nested structure from backend)
      const platonic = geo.platonic_similarities || {};
      const symmetry = geo.symmetry_metrics || {};
      
      // Transform backend data to frontend format
      // Backend returns: platonic_similarities.icosahedron (0-1 similarity)
      // Frontend displays as percentage
      return {
        patterns: [
          { 
            name: 'Icosahedron', 
            score: platonic.icosahedron || 0, 
            count: geo.phi_pattern_count || 0, 
            percentage: (platonic.icosahedron || 0) * 100
          },
          { 
            name: 'Dodecahedron', 
            score: platonic.dodecahedron || 0, 
            count: geo.phi_pattern_count || 0, 
            percentage: (platonic.dodecahedron || 0) * 100
          },
          { 
            name: 'Octahedron', 
            score: platonic.octahedron || 0, 
            count: geo.phi_pattern_count || 0, 
            percentage: (platonic.octahedron || 0) * 100
          },
          { 
            name: 'Golden Ratio (φ)', 
            score: (geo.golden_ratio_percentage || 0) / 100, 
            count: geo.phi_pattern_count || 0, 
            percentage: geo.golden_ratio_percentage || 0 
          },
        ],
        phiAngles: geo.phi_angles || [],
        symmetry: {
          overall: symmetry.rotational || 0,
          local: symmetry.local || 0,
          global: symmetry.radius_of_gyration || 0,
        },
        qcppMetrics: {
          avgQCP: qcpp.qcp_score || 0,
          fieldCoherence: qcpp.resonance_40hz || 0,
          thzPeaks: geo.thz_peaks || [],
          quantumStability: qcpp.qaap_alignment || 0,
          waterShielding: qcpp.water_shielding || 0,
          cacheHitRate: qcpp.qcpp_cache_hit_rate || 0,
        },
        raw: geo,  // Keep raw data for debugging
      };
    },
  });

  const patterns: GeometricPattern[] = geometricData?.patterns.map((p, idx) => ({
    ...p,
    icon: [<Hexagon />, <ChangeHistory />, <Crop169 />, <Hexagon />][idx],
    color: ['#8884d8', '#82ca9d', '#ffc658', '#ff8042'][idx],
  })) || [];

  // Prepare radar chart data
  const radarData = [
    {
      metric: 'Icosahedron',
      score: (geometricData?.patterns[0]?.score || 0) * 100,
    },
    {
      metric: 'Dodecahedron',
      score: (geometricData?.patterns[1]?.score || 0) * 100,
    },
    {
      metric: 'Octahedron',
      score: (geometricData?.patterns[2]?.score || 0) * 100,
    },
    {
      metric: 'Symmetry',
      score: (geometricData?.symmetry?.overall || 0) * 100,
    },
    {
      metric: 'QCP',
      score: ((geometricData?.qcppMetrics?.avgQCP || 0) / 10) * 100,
    },
    {
      metric: 'Coherence',
      score: (geometricData?.qcppMetrics?.fieldCoherence || 0) * 100,
    },
  ];

  // Phi angle distribution
  const phiAngleDistribution = geometricData?.phiAngles.reduce((acc: Record<string, number>, angle: { category: string }) => {
    const category = angle.category;
    acc[category] = (acc[category] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const distributionData = [
    { category: 'Excellent', count: phiAngleDistribution?.excellent || 0, color: '#4caf50' },
    { category: 'Good', count: phiAngleDistribution?.good || 0, color: '#ff9800' },
    { category: 'Poor', count: phiAngleDistribution?.poor || 0, color: '#f44336' },
  ];

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error">
        Failed to load geometric analysis data. {(error as Error).message}
      </Alert>
    );
  }

  if (!geometricData) {
    return (
      <Alert severity="info">
        No geometric analysis data available for this prediction.
      </Alert>
    );
  }

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
          <ToggleButton value="overview">Overview</ToggleButton>
          <ToggleButton value="details">Detailed Analysis</ToggleButton>
        </ToggleButtonGroup>
      </Box>

      {viewMode === 'overview' ? (
        <>
          {/* Geometric Pattern Overview */}
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Geometric Pattern Recognition
            </Typography>
            <Typography variant="caption" color="text.secondary" display="block" mb={3}>
              Presence of Platonic solid geometries in protein structure
            </Typography>

            <Stack spacing={3}>
              {patterns.map((pattern, idx) => (
                <Box key={idx}>
                  <Box display="flex" alignItems="center" gap={2} mb={1}>
                    <Box sx={{ color: pattern.color }}>{pattern.icon}</Box>
                    <Typography variant="body2" flex={1}>
                      {pattern.name}
                    </Typography>
                    <Chip
                      label={`${pattern.count} instances`}
                      size="small"
                      variant="outlined"
                    />
                    <Typography variant="body2" fontWeight="bold" sx={{ minWidth: 60 }}>
                      {(pattern.score * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={pattern.score * 100}
                    sx={{
                      height: 8,
                      borderRadius: 4,
                      backgroundColor: 'rgba(0,0,0,0.1)',
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: pattern.color,
                      },
                    }}
                  />
                </Box>
              ))}
            </Stack>
          </Paper>

          {/* Geometric Metrics Radar */}
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Multi-Dimensional Geometric Profile
            </Typography>
            <Typography variant="caption" color="text.secondary" display="block" mb={2}>
              Comprehensive view of geometric and quantum metrics
            </Typography>
            <ResponsiveContainer width="100%" height={400}>
              <RadarChart data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="metric" />
                <PolarRadiusAxis angle={90} domain={[0, 100]} />
                <Radar
                  name="Geometric Score"
                  dataKey="score"
                  stroke="#8884d8"
                  fill="#8884d8"
                  fillOpacity={0.6}
                />
                <Legend />
                <Tooltip formatter={(value: number) => `${value.toFixed(1)}%`} />
              </RadarChart>
            </ResponsiveContainer>
          </Paper>

          {/* Phi Angle Distribution */}
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Golden Ratio (Φ) Angle Distribution
            </Typography>
            <Typography variant="caption" color="text.secondary" display="block" mb={2}>
              Alignment with phi angle (137.5°) across residues
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={distributionData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" />
                <YAxis label={{ value: 'Count', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="count" name="Residue Count">
                  {distributionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Paper>

          {/* QCPP Metrics Summary */}
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Quantum Coherence Metrics
            </Typography>
            <Divider sx={{ my: 2 }} />
            <Stack spacing={2}>
              <Box display="flex" justifyContent="space-between">
                <Typography variant="body2" color="text.secondary">
                  Average QCP Score
                </Typography>
                <Typography variant="body1" fontWeight="bold">
                  {geometricData.qcppMetrics.avgQCP.toFixed(2)}
                </Typography>
              </Box>
              <Box display="flex" justifyContent="space-between">
                <Typography variant="body2" color="text.secondary">
                  Field Coherence
                </Typography>
                <Typography variant="body1" fontWeight="bold">
                  {(geometricData.qcppMetrics.fieldCoherence * 100).toFixed(1)}%
                </Typography>
              </Box>
              <Box display="flex" justifyContent="space-between">
                <Typography variant="body2" color="text.secondary">
                  Quantum Stability
                </Typography>
                <Typography variant="body1" fontWeight="bold">
                  {(geometricData.qcppMetrics.quantumStability * 100).toFixed(1)}%
                </Typography>
              </Box>
              <Box>
                <Typography variant="body2" color="text.secondary" mb={1}>
                  THz Spectral Peaks
                </Typography>
                <Box display="flex" gap={1} flexWrap="wrap">
                  {geometricData.qcppMetrics.thzPeaks.map((peak: number, idx: number) => (
                    <Chip
                      key={idx}
                      label={`${peak.toFixed(2)} THz`}
                      size="small"
                      variant="outlined"
                    />
                  ))}
                </Box>
              </Box>
            </Stack>
          </Paper>
        </>
      ) : (
        <>
          {/* Detailed Phi Angle Analysis */}
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Detailed Phi Angle Analysis
            </Typography>
            <Typography variant="caption" color="text.secondary" display="block" mb={2}>
              Per-residue alignment with golden ratio angle (137.5°)
            </Typography>
            <TableContainer sx={{ maxHeight: 500 }}>
              <Table size="small" stickyHeader>
                <TableHead>
                  <TableRow>
                    <TableCell><strong>Residue</strong></TableCell>
                    <TableCell align="right"><strong>Angle (°)</strong></TableCell>
                    <TableCell align="right"><strong>Deviation</strong></TableCell>
                    <TableCell align="center"><strong>Quality</strong></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {geometricData.phiAngles.map((angle: { residue: string; angle: number; deviation: number; category: string }, idx: number) => (
                    <TableRow key={idx} hover>
                      <TableCell>{angle.residue}</TableCell>
                      <TableCell align="right">{angle.angle.toFixed(2)}°</TableCell>
                      <TableCell align="right">{angle.deviation.toFixed(2)}°</TableCell>
                      <TableCell align="center">
                        <Chip
                          label={angle.category.toUpperCase()}
                          color={
                            angle.category === 'excellent' ? 'success' :
                            angle.category === 'good' ? 'warning' : 'error'
                          }
                          size="small"
                        />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>

          {/* Symmetry Breakdown */}
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Symmetry Analysis
            </Typography>
            <Divider sx={{ my: 2 }} />
            <Stack spacing={3}>
              <Box>
                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography variant="body2" color="text.secondary">
                    Overall Symmetry
                  </Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {(geometricData.symmetry.overall * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={geometricData.symmetry.overall * 100}
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>

              <Box>
                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography variant="body2" color="text.secondary">
                    Local Symmetry
                  </Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {(geometricData.symmetry.local * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={geometricData.symmetry.local * 100}
                  sx={{ height: 8, borderRadius: 4 }}
                  color="secondary"
                />
              </Box>

              <Box>
                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography variant="body2" color="text.secondary">
                    Global Symmetry
                  </Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {(geometricData.symmetry.global * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={geometricData.symmetry.global * 100}
                  sx={{ height: 8, borderRadius: 4 }}
                  color="success"
                />
              </Box>
            </Stack>
          </Paper>

          {/* Pattern Details Table */}
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Geometric Pattern Details
            </Typography>
            <Divider sx={{ my: 2 }} />
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell><strong>Pattern</strong></TableCell>
                    <TableCell align="right"><strong>Score</strong></TableCell>
                    <TableCell align="right"><strong>Instances</strong></TableCell>
                    <TableCell align="right"><strong>Coverage</strong></TableCell>
                    <TableCell><strong>Quality</strong></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {patterns.map((pattern, idx) => (
                    <TableRow key={idx} hover>
                      <TableCell>
                        <Box display="flex" alignItems="center" gap={1}>
                          <Box sx={{ color: pattern.color }}>{pattern.icon}</Box>
                          {pattern.name}
                        </Box>
                      </TableCell>
                      <TableCell align="right">
                        {(pattern.score * 100).toFixed(1)}%
                      </TableCell>
                      <TableCell align="right">{pattern.count}</TableCell>
                      <TableCell align="right">{pattern.percentage.toFixed(1)}%</TableCell>
                      <TableCell>
                        <Chip
                          label={pattern.score > 0.7 ? 'High' : pattern.score > 0.5 ? 'Medium' : 'Low'}
                          color={pattern.score > 0.7 ? 'success' : pattern.score > 0.5 ? 'warning' : 'error'}
                          size="small"
                        />
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

export default GeometricAnalysisTab;
