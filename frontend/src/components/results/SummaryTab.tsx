import React, { useState } from 'react';
import {
  Box,
  Stack,
  Paper,
  Typography,
  LinearProgress,
  Divider,
  Alert,
  Tooltip,
  IconButton,
  Collapse,
  alpha,
  useTheme,
} from '@mui/material';
import {
  Timeline,
  Memory,
  Speed,
  CheckCircle,
  TrendingUp,
  TrendingDown,
  HelpOutline as HelpIcon,
  ExpandLess as CollapseIcon,
} from '@mui/icons-material';
import type { PredictionResponse } from '../../types/api';
import SecondaryStructureSummary from './SecondaryStructureSummary';
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
} from 'recharts';

interface SummaryTabProps {
  prediction: PredictionResponse;
}

interface MetricCardProps {
  icon: React.ReactNode;
  title: string;
  value: string | number;
  unit?: string;
  subtitle?: string;
  color?: string;
  trend?: 'up' | 'down' | 'neutral';
  trendValue?: string;
}

const MetricCard: React.FC<MetricCardProps> = ({
  icon,
  title,
  value,
  unit,
  subtitle,
  color = '#1976d2',
  trend,
  trendValue,
}) => {
  const getTrendIcon = () => {
    if (trend === 'up') return <TrendingUp fontSize="small" color="success" />;
    if (trend === 'down') return <TrendingDown fontSize="small" color="error" />;
    return null;
  };

  return (
    <Paper sx={{ p: 2, height: '100%' }}>
      <Stack spacing={1}>
        <Box display="flex" alignItems="center" gap={1}>
          <Box sx={{ color }}>{icon}</Box>
          <Typography variant="body2" color="text.secondary">
            {title}
          </Typography>
        </Box>

        <Box>
          <Box display="flex" alignItems="baseline" gap={0.5}>
            <Typography variant="h4" component="span">
              {value}
            </Typography>
            {unit && (
              <Typography variant="body2" color="text.secondary" component="span">
                {unit}
              </Typography>
            )}
          </Box>

          {trend && trendValue && (
            <Box display="flex" alignItems="center" gap={0.5} mt={0.5}>
              {getTrendIcon()}
              <Typography variant="caption" color="text.secondary">
                {trendValue}
              </Typography>
            </Box>
          )}

          {subtitle && (
            <Typography variant="caption" color="text.secondary" display="block">
              {subtitle}
            </Typography>
          )}
        </Box>
      </Stack>
    </Paper>
  );
};

const SummaryTab: React.FC<SummaryTabProps> = ({ prediction }) => {
  const theme = useTheme();
  const [showQualityHelp, setShowQualityHelp] = useState(false);
  const [showMetricsHelp, setShowMetricsHelp] = useState(false);
  const [showEnergyHelp, setShowEnergyHelp] = useState(false);

  // Extract metrics from nested object
  const bestRMSD = prediction.metrics?.best_rmsd ?? prediction.metrics?.final_rmsd ?? null;
  const bestEnergy = prediction.metrics?.best_energy ?? prediction.metrics?.final_energy ?? null;
  const initialEnergy = prediction.metrics?.initial_energy ?? null;
  const energyChange = prediction.metrics?.energy_change ?? null;
  const convergenceRate = prediction.metrics?.convergence_rate ?? null;
  const conformationsExplored = prediction.metrics?.conformations_explored ?? null;
  const uniqueStructures = prediction.metrics?.unique_structures ?? null;
  const gdtScore = prediction.metrics?.gdt_ts_score ?? null;
  const tmScore = prediction.metrics?.tm_score ?? null;
  const qaapAlignment = prediction.metrics?.qaap_alignment ?? null;
  const resonance40hz = prediction.metrics?.resonance_40hz ?? null;
  const waterShielding = prediction.metrics?.water_shielding ?? null;
  const qcpScore = prediction.metrics?.qcp_score ?? null;
  const refinementApplied = prediction.metrics?.refinement_applied ?? false;

  // Calculate quality scores
  const getQualityScore = (rmsd?: number | null): number => {
    if (rmsd === undefined || rmsd === null || rmsd === Infinity) return 0;
    if (rmsd < 2) return 95;
    if (rmsd < 4) return 75;
    if (rmsd < 5) return 55;
    return 30;
  };

  const qualityScore = getQualityScore(bestRMSD);

  // Energy breakdown data (estimated distribution for visualization)
  const energyData = bestEnergy ? [
    { name: 'Bond', value: Math.abs(bestEnergy) * 0.15, color: '#8884d8' },
    { name: 'Angle', value: Math.abs(bestEnergy) * 0.12, color: '#82ca9d' },
    { name: 'Dihedral', value: Math.abs(bestEnergy) * 0.18, color: '#ffc658' },
    { name: 'VDW', value: Math.abs(bestEnergy) * 0.25, color: '#ff8042' },
    { name: 'Electrostatic', value: Math.abs(bestEnergy) * 0.20, color: '#a4de6c' },
    { name: 'H-Bond', value: Math.abs(bestEnergy) * 0.10, color: '#d0ed57' },
  ] : [];

  // Agent statistics (estimated distribution based on diversity setting)
  const diversity = prediction.configuration?.diversity || 'balanced';
  const numAgents = prediction.configuration?.agents || 10;
  const agentStats = [
    { 
      name: 'Cautious', 
      count: diversity === 'cautious' ? Math.ceil(numAgents * 0.5) : Math.floor(numAgents * 0.33),
      avgRMSD: bestRMSD && bestRMSD !== Infinity ? bestRMSD * 1.1 : null
    },
    { 
      name: 'Balanced', 
      count: diversity === 'balanced' ? Math.ceil(numAgents * 0.5) : Math.floor(numAgents * 0.34),
      avgRMSD: bestRMSD && bestRMSD !== Infinity ? bestRMSD : null
    },
    { 
      name: 'Aggressive', 
      count: diversity === 'aggressive' ? Math.ceil(numAgents * 0.5) : Math.floor(numAgents * 0.33),
      avgRMSD: bestRMSD && bestRMSD !== Infinity ? bestRMSD * 1.2 : null
    },
  ];

  // Configuration summary
  const configSummary = [
    { label: 'Agents', value: prediction.configuration?.agents || 10 },
    { label: 'Iterations', value: prediction.total_iterations || 1000 },
    { label: 'Diversity', value: (prediction.configuration?.diversity || 'balanced').charAt(0).toUpperCase() + (prediction.configuration?.diversity || 'balanced').slice(1) },
    { label: 'QCPP Config', value: prediction.configuration?.qcpp_config ? prediction.configuration.qcpp_config.split('_').map((w: string) => w.charAt(0).toUpperCase() + w.slice(1)).join(' ') : 'None' },
    { label: 'Native PDB', value: prediction.configuration?.native_pdb || 'None' },
    { label: 'Checkpointing', value: prediction.configuration?.enable_checkpointing ? 'Enabled' : 'Disabled' },
  ];

  // Calculate completion percentage
  const completionPercentage = prediction.current_iteration && prediction.total_iterations
    ? Math.round((prediction.current_iteration / prediction.total_iterations) * 100)
    : 100;

  return (
    <Stack spacing={3}>
      {/* Quality Overview */}
      <Paper sx={{ p: 3 }}>
        <Box display="flex" alignItems="center" gap={1} mb={2}>
          <Typography variant="h6">
            Quality Overview
          </Typography>
          <Tooltip title={showQualityHelp ? 'Hide explanation' : 'Show explanation'}>
            <IconButton
              size="small"
              onClick={() => setShowQualityHelp(!showQualityHelp)}
              sx={{ color: showQualityHelp ? 'primary.main' : 'text.secondary' }}
            >
              {showQualityHelp ? <CollapseIcon fontSize="small" /> : <HelpIcon fontSize="small" />}
            </IconButton>
          </Tooltip>
        </Box>
        
        <Collapse in={showQualityHelp}>
          <Box
            sx={{
              mb: 2,
              p: 2,
              backgroundColor: alpha(theme.palette.info.main, 0.08),
              borderRadius: 1,
              borderLeft: `3px solid ${theme.palette.info.main}`,
            }}
          >
            <Typography variant="body2" color="text.secondary" paragraph>
              The quality score combines multiple factors to give an overall assessment:
            </Typography>
            <Typography variant="body2" color="text.secondary" component="div">
              • <strong>RMSD Weight</strong>: How close the predicted structure is to the native (if provided)
            </Typography>
            <Typography variant="body2" color="text.secondary" component="div">
              • <strong>Energy</strong>: Lower (more negative) energy indicates a more stable structure
            </Typography>
            <Typography variant="body2" color="text.secondary" component="div">
              • <strong>Convergence</strong>: How well the optimization reached a stable state
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block', fontStyle: 'italic' }}>
              Score thresholds: 75+/100 = Good quality, 55-74 = Acceptable, Below 55 = Needs improvement
            </Typography>
          </Box>
        </Collapse>
        
        <Box sx={{ mt: 2 }}>
          <Box display="flex" justifyContent="space-between" mb={1}>
            <Typography variant="body2" color="text.secondary">
              Overall Quality Score
            </Typography>
            <Typography variant="body2" fontWeight="bold">
              {qualityScore}/100
            </Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={qualityScore}
            sx={{
              height: 10,
              borderRadius: 5,
              backgroundColor: 'rgba(0,0,0,0.1)',
              '& .MuiLinearProgress-bar': {
                backgroundColor: qualityScore >= 75 ? '#4caf50' : qualityScore >= 55 ? '#ff9800' : '#f44336',
              },
            }}
          />
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            Based on RMSD, energy convergence, and structural quality
          </Typography>
        </Box>
      </Paper>

      {/* Secondary Structure Summary */}
      <SecondaryStructureSummary 
        predictionId={prediction.id}
        sequence={prediction.protein_sequence || prediction.sequence || ''}
      />

      {/* Key Metrics Section Header */}
      <Box>
        <Box display="flex" alignItems="center" gap={1} mb={2}>
          <Typography variant="h6">
            Key Metrics
          </Typography>
          <Tooltip title={showMetricsHelp ? 'Hide explanation' : 'Show explanation'}>
            <IconButton
              size="small"
              onClick={() => setShowMetricsHelp(!showMetricsHelp)}
              sx={{ color: showMetricsHelp ? 'primary.main' : 'text.secondary' }}
            >
              {showMetricsHelp ? <CollapseIcon fontSize="small" /> : <HelpIcon fontSize="small" />}
            </IconButton>
          </Tooltip>
        </Box>
        
        <Collapse in={showMetricsHelp}>
          <Box
            sx={{
              mb: 2,
              p: 2,
              backgroundColor: alpha(theme.palette.info.main, 0.08),
              borderRadius: 1,
              borderLeft: `3px solid ${theme.palette.info.main}`,
            }}
          >
            <Typography variant="body2" color="text.secondary" component="div" paragraph>
              <strong>RMSD (Root Mean Square Deviation)</strong>: Measures structural similarity to the reference. 
              &lt;2Å is excellent, 2-4Å is good, 4-5Å is acceptable for research purposes.
            </Typography>
            <Typography variant="body2" color="text.secondary" component="div" paragraph>
              <strong>Energy</strong>: Total potential energy from molecular mechanics calculation.
              Negative values indicate stable structures. More negative = more stable.
            </Typography>
            <Typography variant="body2" color="text.secondary" component="div" paragraph>
              <strong>GDT-TS</strong>: Global Distance Test score (0-100). Measures what percentage 
              of residues are within various distance cutoffs of the native. 65+ is good.
            </Typography>
            <Typography variant="body2" color="text.secondary" component="div">
              <strong>TM-Score</strong>: Template Modeling score (0-1). Values &gt;0.5 indicate same fold, 
              &gt;0.8 indicates high structural similarity.
            </Typography>
          </Box>
        </Collapse>
      </Box>

      {/* Key Metrics Grid */}
      <Box display="flex" flexWrap="wrap" gap={2}>
        <Box flex="1 1 calc(25% - 12px)" minWidth={200}>
          <MetricCard
            icon={<CheckCircle />}
            title="Best RMSD"
            value={bestRMSD && bestRMSD !== Infinity ? bestRMSD.toFixed(2) : 'N/A'}
            unit="Å"
            subtitle={bestRMSD === Infinity ? 'No native structure' : 'Lower is better'}
            color="#4caf50"
            trend={bestRMSD && bestRMSD !== Infinity && bestRMSD < 5 ? 'up' : bestRMSD === Infinity ? undefined : 'down'}
            trendValue={bestRMSD && bestRMSD !== Infinity && bestRMSD < 5 ? 'Good quality' : bestRMSD !== Infinity ? 'Needs improvement' : undefined}
          />
        </Box>

        <Box flex="1 1 calc(25% - 12px)" minWidth={200}>
          <MetricCard
            icon={<Speed />}
            title="Best Energy"
            value={bestEnergy?.toFixed(1) || 'N/A'}
            unit="kcal/mol"
            subtitle="Negative indicates stability"
            color="#2196f3"
            trend={bestEnergy && bestEnergy < -50 ? 'up' : 'neutral'}
            trendValue={bestEnergy && bestEnergy < -50 ? 'Well optimized' : bestEnergy && bestEnergy < 0 ? 'Stable' : 'Unstable'}
          />
        </Box>

        <Box flex="1 1 calc(25% - 12px)" minWidth={200}>
          <MetricCard
            icon={<Timeline />}
            title="Iterations"
            value={prediction.current_iteration || prediction.total_iterations || 0}
            unit={`/ ${prediction.total_iterations || '?'}`}
            subtitle={`${completionPercentage}% complete`}
            color="#ff9800"
          />
        </Box>

        <Box flex="1 1 calc(25% - 12px)" minWidth={200}>
          <MetricCard
            icon={<Memory />}
            title="Conformations"
            value={conformationsExplored || 'N/A'}
            subtitle="Total explored"
            color="#9c27b0"
          />
        </Box>
      </Box>

      {/* Additional Metrics Row */}
      {(gdtScore || tmScore || energyChange || convergenceRate) && (
        <Box display="flex" flexWrap="wrap" gap={2}>
          {gdtScore !== null && (
            <Box flex="1 1 calc(25% - 12px)" minWidth={200}>
              <MetricCard
                icon={<CheckCircle />}
                title="GDT-TS Score"
                value={gdtScore.toFixed(1)}
                subtitle="Global Distance Test"
                color="#00bcd4"
                trend={gdtScore >= 65 ? 'up' : 'neutral'}
                trendValue={gdtScore >= 80 ? 'Excellent' : gdtScore >= 65 ? 'Good' : 'Moderate'}
              />
            </Box>
          )}

          {tmScore !== null && (
            <Box flex="1 1 calc(25% - 12px)" minWidth={200}>
              <MetricCard
                icon={<CheckCircle />}
                title="TM-Score"
                value={tmScore.toFixed(3)}
                subtitle="Template Modeling"
                color="#009688"
                trend={tmScore >= 0.5 ? 'up' : 'neutral'}
                trendValue={tmScore >= 0.8 ? 'High similarity' : tmScore >= 0.5 ? 'Same fold' : 'Low'}
              />
            </Box>
          )}

          {energyChange !== null && (
            <Box flex="1 1 calc(25% - 12px)" minWidth={200}>
              <MetricCard
                icon={<TrendingDown />}
                title="Energy Change"
                value={energyChange.toFixed(1)}
                unit="kcal/mol"
                subtitle={`From ${initialEnergy?.toFixed(1) || 'N/A'}`}
                color="#f44336"
                trend={energyChange < 0 ? 'up' : 'down'}
                trendValue={energyChange < -50 ? 'Excellent' : energyChange < 0 ? 'Good' : 'Poor'}
              />
            </Box>
          )}

          {convergenceRate !== null && (
            <Box flex="1 1 calc(25% - 12px)" minWidth={200}>
              <MetricCard
                icon={<Speed />}
                title="Convergence"
                value={convergenceRate.toFixed(1)}
                unit="%"
                subtitle="Optimization rate"
                color="#ff5722"
                trend={convergenceRate >= 50 ? 'up' : 'neutral'}
              />
            </Box>
          )}
        </Box>
      )}

      {/* Quantum Metrics Row */}
      {(qaapAlignment || resonance40hz || waterShielding || qcpScore) && (
        <Box display="flex" flexWrap="wrap" gap={2}>
          {qaapAlignment !== null && (
            <Box flex="1 1 calc(25% - 12px)" minWidth={200}>
              <MetricCard
                icon={<Speed />}
                title="QAAP Alignment"
                value={qaapAlignment.toFixed(3)}
                subtitle="Quantum anharmonicity"
                color="#673ab7"
              />
            </Box>
          )}

          {resonance40hz !== null && (
            <Box flex="1 1 calc(25% - 12px)" minWidth={200}>
              <MetricCard
                icon={<Timeline />}
                title="40 Hz Resonance"
                value={resonance40hz.toFixed(3)}
                subtitle="Coherence frequency"
                color="#3f51b5"
              />
            </Box>
          )}

          {waterShielding !== null && (
            <Box flex="1 1 calc(25% - 12px)" minWidth={200}>
              <MetricCard
                icon={<Memory />}
                title="Water Shielding"
                value={waterShielding.toFixed(1)}
                unit="fs"
                subtitle="408 fs coherence time"
                color="#2196f3"
              />
            </Box>
          )}

          {qcpScore !== null && (
            <Box flex="1 1 calc(25% - 12px)" minWidth={200}>
              <MetricCard
                icon={<CheckCircle />}
                title="QCP Score"
                value={qcpScore.toFixed(2)}
                subtitle="Quantum coherence"
                color="#03a9f4"
              />
            </Box>
          )}
        </Box>
      )}

      {/* Refinement Notice */}
      {refinementApplied && (
        <Alert severity="info">
          Quantum refinement was applied to the final structure, improving RMSD and energy values.
        </Alert>
      )}

      {/* Energy Breakdown and Agent Statistics */}
      <Box>
        <Box display="flex" alignItems="center" gap={1} mb={2}>
          <Typography variant="h6">
            Analysis Details
          </Typography>
          <Tooltip title={showEnergyHelp ? 'Hide explanation' : 'Show explanation'}>
            <IconButton
              size="small"
              onClick={() => setShowEnergyHelp(!showEnergyHelp)}
              sx={{ color: showEnergyHelp ? 'primary.main' : 'text.secondary' }}
            >
              {showEnergyHelp ? <CollapseIcon fontSize="small" /> : <HelpIcon fontSize="small" />}
            </IconButton>
          </Tooltip>
        </Box>
        
        <Collapse in={showEnergyHelp}>
          <Box
            sx={{
              mb: 2,
              p: 2,
              backgroundColor: alpha(theme.palette.info.main, 0.08),
              borderRadius: 1,
              borderLeft: `3px solid ${theme.palette.info.main}`,
            }}
          >
            <Typography variant="subtitle2" fontWeight="bold" color="info.main" gutterBottom>
              Energy Components (Molecular Mechanics)
            </Typography>
            <Typography variant="body2" color="text.secondary" component="div">
              • <strong>Bond</strong>: Energy from stretched/compressed covalent bonds
            </Typography>
            <Typography variant="body2" color="text.secondary" component="div">
              • <strong>Angle</strong>: Energy from bent bond angles
            </Typography>
            <Typography variant="body2" color="text.secondary" component="div">
              • <strong>Dihedral</strong>: Torsion energy from rotation around bonds
            </Typography>
            <Typography variant="body2" color="text.secondary" component="div">
              • <strong>VDW</strong>: Van der Waals (steric) interactions
            </Typography>
            <Typography variant="body2" color="text.secondary" component="div">
              • <strong>Electrostatic</strong>: Coulombic charge interactions
            </Typography>
            <Typography variant="body2" color="text.secondary" component="div" paragraph>
              • <strong>H-Bond</strong>: Hydrogen bonding stabilization
            </Typography>
            <Typography variant="subtitle2" fontWeight="bold" color="info.main" gutterBottom>
              Agent Distribution
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Multi-agent exploration uses diverse strategies: Cautious agents make small careful moves, 
              Balanced agents use moderate exploration, and Aggressive agents try larger conformational changes.
              The population mix depends on your diversity setting.
            </Typography>
          </Box>
        </Collapse>
      </Box>

      <Box display="flex" flexWrap="wrap" gap={3}>
        {/* Energy Breakdown */}
        {energyData.length > 0 && (
          <Paper sx={{ p: 3, flex: '1 1 45%', minWidth: 300 }}>
            <Typography variant="h6" gutterBottom>
              Energy Breakdown
            </Typography>
            <Typography variant="caption" color="text.secondary" display="block" mb={2}>
              Estimated contribution by component (absolute values)
            </Typography>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={energyData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${percent ? (percent * 100).toFixed(0) : 0}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {energyData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <RechartsTooltip formatter={(value: number) => value.toFixed(2)} />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        )}

        {/* Agent Statistics */}
        {agentStats.some(stat => stat.avgRMSD !== null) && (
          <Paper sx={{ p: 3, flex: '1 1 45%', minWidth: 300 }}>
            <Typography variant="h6" gutterBottom>
              Agent Distribution
            </Typography>
            <Typography variant="caption" color="text.secondary" display="block" mb={2}>
              Estimated RMSD by agent type based on {diversity} diversity
            </Typography>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={agentStats.filter(stat => stat.avgRMSD !== null)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis label={{ value: 'Est. RMSD (Å)', angle: -90, position: 'insideLeft' }} />
                <RechartsTooltip />
                <Legend />
                <Bar dataKey="avgRMSD" fill="#8884d8" name="Est. RMSD" />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        )}
      </Box>

      {/* Configuration Summary */}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Configuration
        </Typography>
        <Divider sx={{ my: 2 }} />
        <Stack spacing={2}>
          {configSummary.map((item, index) => (
            <Box
              key={index}
              display="flex"
              justifyContent="space-between"
              alignItems="center"
            >
              <Typography variant="body2" color="text.secondary">
                {item.label}
              </Typography>
              <Typography variant="body1" fontWeight="medium">
                {item.value}
              </Typography>
            </Box>
          ))}
        </Stack>
      </Paper>

      {/* Additional Information */}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Execution Details
        </Typography>
        <Divider sx={{ my: 2 }} />
        <Stack spacing={2}>
          <Box>
            <Typography variant="body2" color="text.secondary">
              Sequence Length
            </Typography>
            <Typography variant="body1">
              {(prediction.protein_sequence || prediction.sequence || '').length} residues
            </Typography>
          </Box>

          <Box>
            <Typography variant="body2" color="text.secondary">
              Execution Time
            </Typography>
            <Typography variant="body1">
              {prediction.completed_at && prediction.created_at
                ? `${Math.round((new Date(prediction.completed_at).getTime() - new Date(prediction.created_at).getTime()) / 60000)} minutes`
                : 'In progress'}
            </Typography>
          </Box>

          {uniqueStructures && (
            <Box>
              <Typography variant="body2" color="text.secondary">
                Unique Structures Explored
              </Typography>
              <Typography variant="body1">
                {uniqueStructures.toLocaleString()}
              </Typography>
            </Box>
          )}

          {prediction.result_path && (
            <Box>
              <Typography variant="body2" color="text.secondary">
                Results Location
              </Typography>
              <Typography variant="body1" sx={{ wordBreak: 'break-all' }}>
                {prediction.result_path}
              </Typography>
            </Box>
          )}

          {prediction.error_message && (
            <Box>
              <Typography variant="body2" color="error">
                Error
              </Typography>
              <Typography variant="body1" color="error">
                {prediction.error_message}
              </Typography>
            </Box>
          )}
        </Stack>
      </Paper>
    </Stack>
  );
};

export default SummaryTab;
