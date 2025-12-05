import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Box,
  Stack,
  Paper,
  Typography,
  Slider,
  Button,
  IconButton,
  Chip,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
  CircularProgress,
  type SelectChangeEvent,
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  SkipPrevious,
  SkipNext,
  Download,
  FilterList,
} from '@mui/icons-material';
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Cell,
} from 'recharts';
import api from '../../services/api';

interface TrajectoryTabProps {
  predictionId: string;
}

interface TrajectoryPoint {
  iteration: number;
  energy: number;
  rmsd: number;
  aggressiveness: number;
  consistency: number;
  agent_id?: number;
  is_best?: boolean;
}

const TrajectoryTab: React.FC<TrajectoryTabProps> = ({ predictionId }) => {
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [colorBy, setColorBy] = useState<'energy' | 'agent' | 'rmsd'>('energy');
  const [filterAgent, setFilterAgent] = useState<number | 'all'>('all');

  // Fetch trajectory data from backend
  const { data: trajectory, isLoading, error } = useQuery({
    queryKey: ['trajectory', predictionId],
    queryFn: async () => {
      const response = await api.get(`/results/${predictionId}/trajectory`);
      const data = response.data;
      
      // Transform backend data to frontend format
      if (!data.trajectory || data.trajectory.length === 0) {
        return [];
      }
      
      // Helper to parse agent_id from various formats
      const parseAgentId = (agentId: string | number): number => {
        if (typeof agentId === 'number') return agentId;
        if (typeof agentId === 'string') {
          // Handle formats: "agent_0", "agent_1", "0", "1", etc.
          const match = agentId.match(/\d+/);
          return match ? parseInt(match[0], 10) : 0;
        }
        return 0;
      };
      
      // Track the best energy point to mark as is_best
      let bestEnergy = Infinity;
      let bestIndex = -1;
      data.trajectory.forEach((point: { energy: number }, index: number) => {
        if (point.energy < bestEnergy) {
          bestEnergy = point.energy;
          bestIndex = index;
        }
      });
      
      const points: TrajectoryPoint[] = data.trajectory.map((point: {
        iteration: number;
        energy: number;
        rmsd?: number | null;
        aggressiveness: number;
        consistency: number;
        agent_id: string | number;
        is_best?: boolean;
      }, index: number) => ({
        iteration: point.iteration,
        energy: point.energy,
        rmsd: point.rmsd ?? 0,
        aggressiveness: point.aggressiveness,
        consistency: point.consistency,
        agent_id: parseAgentId(point.agent_id),
        is_best: point.is_best ?? (index === bestIndex),
      }));
      
      return points;
    },
  });

  // Playback control
  React.useEffect(() => {
    if (!isPlaying || !trajectory) return;

    const interval = setInterval(() => {
      setCurrentFrame((prev) => {
        if (prev >= trajectory.length - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 100);

    return () => clearInterval(interval);
  }, [isPlaying, trajectory]);

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const handlePrevious = () => {
    setCurrentFrame((prev) => Math.max(0, prev - 1));
    setIsPlaying(false);
  };

  const handleNext = () => {
    if (!trajectory) return;
    setCurrentFrame((prev) => Math.min(trajectory.length - 1, prev + 1));
    setIsPlaying(false);
  };

  const handleSliderChange = (_event: Event, value: number | number[]) => {
    setCurrentFrame(value as number);
    setIsPlaying(false);
  };

  const handleColorByChange = (event: SelectChangeEvent<'energy' | 'agent' | 'rmsd'>) => {
    setColorBy(event.target.value as 'energy' | 'agent' | 'rmsd');
  };

  const handleFilterChange = (event: SelectChangeEvent<number | 'all'>) => {
    setFilterAgent(event.target.value as number | 'all');
  };

  const getPointColor = (point: TrajectoryPoint) => {
    if (point.is_best) return '#ffd700'; // Gold for best structures

    switch (colorBy) {
      case 'energy': {
        const normalized = Math.max(0, Math.min(1, (point.energy + 200) / 200));
        const r = Math.floor(255 * (1 - normalized));
        const g = Math.floor(255 * normalized);
        return `rgb(${r}, ${g}, 100)`;
      }
      case 'rmsd': {
        const normalized = Math.max(0, Math.min(1, (15 - point.rmsd) / 15));
        const r = Math.floor(255 * (1 - normalized));
        const g = Math.floor(255 * normalized);
        return `rgb(${r}, ${g}, 100)`;
      }
      case 'agent': {
        const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#a4de6c', '#d0ed57', '#8dd1e1', '#d084d0', '#82ca82', '#ffa042'];
        return colors[point.agent_id || 0];
      }
    }
  };

  const filteredTrajectory = trajectory?.filter((point) => 
    filterAgent === 'all' || point.agent_id === filterAgent
  ) || [];

  const currentPoint = trajectory?.[currentFrame];

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
        Failed to load trajectory data. {(error as Error).message}
      </Alert>
    );
  }

  if (!trajectory || trajectory.length === 0) {
    return (
      <Alert severity="info">
        No trajectory data available for this prediction.
      </Alert>
    );
  }

  return (
    <Stack spacing={3}>
      {/* Controls */}
      <Paper sx={{ p: 3 }}>
        <Stack spacing={3}>
          {/* Playback Controls */}
          <Box display="flex" alignItems="center" gap={2}>
            <IconButton onClick={handlePrevious} disabled={currentFrame === 0}>
              <SkipPrevious />
            </IconButton>
            <IconButton onClick={handlePlayPause}>
              {isPlaying ? <Pause /> : <PlayArrow />}
            </IconButton>
            <IconButton onClick={handleNext} disabled={currentFrame === trajectory.length - 1}>
              <SkipNext />
            </IconButton>

            <Box flex={1} px={2}>
              <Slider
                value={currentFrame}
                onChange={handleSliderChange}
                min={0}
                max={trajectory.length - 1}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => `Iteration ${trajectory[value].iteration}`}
              />
            </Box>

            <Typography variant="body2" sx={{ minWidth: 120 }}>
              Frame {currentFrame + 1} / {trajectory.length}
            </Typography>
          </Box>

          {/* Filter Controls */}
          <Box display="flex" gap={2}>
            <FormControl size="small" sx={{ minWidth: 150 }}>
              <InputLabel>Color By</InputLabel>
              <Select value={colorBy} onChange={handleColorByChange} label="Color By">
                <MenuItem value="energy">Energy</MenuItem>
                <MenuItem value="rmsd">RMSD</MenuItem>
                <MenuItem value="agent">Agent</MenuItem>
              </Select>
            </FormControl>

            <FormControl size="small" sx={{ minWidth: 150 }}>
              <InputLabel>Filter Agent</InputLabel>
              <Select value={filterAgent} onChange={handleFilterChange} label="Filter Agent">
                <MenuItem value="all">All Agents</MenuItem>
                {Array.from({ length: 10 }, (_, i) => (
                  <MenuItem key={i} value={i}>Agent {i}</MenuItem>
                ))}
              </Select>
            </FormControl>

            <Button
              variant="outlined"
              startIcon={<Download />}
              onClick={() => window.open(`/api/v1/predictions/${predictionId}/download/trajectory`, '_blank')}
            >
              Export Trajectory
            </Button>

            <Chip
              icon={<FilterList />}
              label={`${filteredTrajectory.length} points`}
              size="small"
            />
          </Box>
        </Stack>
      </Paper>

      {/* Current Frame Info */}
      {currentPoint && (
        <Paper sx={{ p: 2 }}>
          <Box display="flex" justifyContent="space-around" alignItems="center" flexWrap="wrap" gap={2}>
            <Box textAlign="center">
              <Typography variant="caption" color="text.secondary">
                Iteration
              </Typography>
              <Typography variant="h6">{currentPoint.iteration}</Typography>
            </Box>
            <Box textAlign="center">
              <Typography variant="caption" color="text.secondary">
                Energy
              </Typography>
              <Typography variant="h6">{currentPoint.energy.toFixed(2)} kcal/mol</Typography>
            </Box>
            <Box textAlign="center">
              <Typography variant="caption" color="text.secondary">
                RMSD
              </Typography>
              <Typography variant="h6">{currentPoint.rmsd.toFixed(3)} Å</Typography>
            </Box>
            <Box textAlign="center">
              <Typography variant="caption" color="text.secondary">
                Aggressiveness
              </Typography>
              <Typography variant="h6">{currentPoint.aggressiveness.toFixed(2)}</Typography>
            </Box>
            <Box textAlign="center">
              <Typography variant="caption" color="text.secondary">
                Consistency
              </Typography>
              <Typography variant="h6">{currentPoint.consistency.toFixed(2)}</Typography>
            </Box>
            {currentPoint.agent_id !== undefined && (
              <Box textAlign="center">
                <Typography variant="caption" color="text.secondary">
                  Agent
                </Typography>
                <Typography variant="h6">#{currentPoint.agent_id}</Typography>
              </Box>
            )}
            {currentPoint.is_best && (
              <Chip label="Best Structure" color="warning" size="small" />
            )}
          </Box>
        </Paper>
      )}

      {/* Energy Landscape (2D Projection) */}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Energy Landscape
        </Typography>
        <Typography variant="caption" color="text.secondary" display="block" mb={2}>
          Conformational space exploration (Energy vs RMSD)
        </Typography>
        <ResponsiveContainer width="100%" height={400}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              type="number"
              dataKey="rmsd"
              name="RMSD"
              unit=" Å"
              label={{ value: 'RMSD (Å)', position: 'insideBottom', offset: -5 }}
            />
            <YAxis
              type="number"
              dataKey="energy"
              name="Energy"
              unit=" kcal/mol"
              label={{ value: 'Energy (kcal/mol)', angle: -90, position: 'insideLeft' }}
            />
            <ZAxis type="number" range={[50, 200]} />
            <Tooltip
              cursor={{ strokeDasharray: '3 3' }}
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const point = payload[0].payload as TrajectoryPoint;
                  return (
                    <Paper sx={{ p: 1.5 }}>
                      <Typography variant="caption" display="block">
                        <strong>Iteration:</strong> {point.iteration}
                      </Typography>
                      <Typography variant="caption" display="block">
                        <strong>Energy:</strong> {point.energy.toFixed(2)} kcal/mol
                      </Typography>
                      <Typography variant="caption" display="block">
                        <strong>RMSD:</strong> {point.rmsd.toFixed(3)} Å
                      </Typography>
                      {point.agent_id !== undefined && (
                        <Typography variant="caption" display="block">
                          <strong>Agent:</strong> #{point.agent_id}
                        </Typography>
                      )}
                    </Paper>
                  );
                }
                return null;
              }}
            />
            <Legend />
            <Scatter name="Conformations" data={filteredTrajectory} fill="#8884d8">
              {filteredTrajectory.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={getPointColor(entry)}
                  opacity={index === currentFrame ? 1 : 0.6}
                  stroke={index === currentFrame ? '#000' : 'none'}
                  strokeWidth={index === currentFrame ? 2 : 0}
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </Paper>

      {/* Exploration Parameter Space */}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Exploration Parameter Space
        </Typography>
        <Typography variant="caption" color="text.secondary" display="block" mb={2}>
          Agent behavior evolution (Aggressiveness vs Consistency)
        </Typography>
        <ResponsiveContainer width="100%" height={400}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              type="number"
              dataKey="aggressiveness"
              name="Aggressiveness"
              domain={[3, 15]}
              label={{ value: 'Aggressiveness', position: 'insideBottom', offset: -5 }}
            />
            <YAxis
              type="number"
              dataKey="consistency"
              name="Consistency"
              domain={[0.2, 1.0]}
              label={{ value: 'Consistency', angle: -90, position: 'insideLeft' }}
            />
            <ZAxis type="number" range={[50, 200]} />
            <Tooltip
              cursor={{ strokeDasharray: '3 3' }}
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const point = payload[0].payload as TrajectoryPoint;
                  return (
                    <Paper sx={{ p: 1.5 }}>
                      <Typography variant="caption" display="block">
                        <strong>Iteration:</strong> {point.iteration}
                      </Typography>
                      <Typography variant="caption" display="block">
                        <strong>Aggressiveness:</strong> {point.aggressiveness.toFixed(2)}
                      </Typography>
                      <Typography variant="caption" display="block">
                        <strong>Consistency:</strong> {point.consistency.toFixed(2)}
                      </Typography>
                      <Typography variant="caption" display="block">
                        <strong>RMSD:</strong> {point.rmsd.toFixed(3)} Å
                      </Typography>
                    </Paper>
                  );
                }
                return null;
              }}
            />
            <Legend />
            <Scatter name="Exploration States" data={filteredTrajectory} fill="#82ca9d">
              {filteredTrajectory.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={getPointColor(entry)}
                  opacity={index === currentFrame ? 1 : 0.6}
                  stroke={index === currentFrame ? '#000' : 'none'}
                  strokeWidth={index === currentFrame ? 2 : 0}
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </Paper>
    </Stack>
  );
};

export default TrajectoryTab;
