import React, { useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  LinearProgress,
  Chip,
  Stack,
  Tooltip,
  Divider,
  useTheme,
  alpha,
} from '@mui/material';
import {
  TrendingUp as HelixIcon,
  ViewStream as SheetIcon,
  Waves as CoilIcon,
  Science as ScienceIcon,
} from '@mui/icons-material';

interface SecondaryStructureData {
  assignments?: string;  // "HHHCCCEEE" format
  helix_count?: number;
  sheet_count?: number;
  coil_count?: number;
  helix_percent?: number;
  sheet_percent?: number;
  coil_percent?: number;
  helix_segments?: [number, number][];
  sheet_segments?: [number, number][];
  coil_segments?: [number, number][];
  total_residues?: number;
}

interface SecondaryStructurePanelProps {
  sequence: string;
  secondaryStructure?: SecondaryStructureData;
  isLoading?: boolean;
  source?: 'structure' | 'sequence_estimate' | 'live';
}

// Colors for secondary structure
const SS_COLORS = {
  H: '#e91e63', // Pink/magenta for helix
  E: '#2196f3', // Blue for sheet
  C: '#9e9e9e', // Gray for coil
};

const SS_LABELS = {
  H: 'α-Helix',
  E: 'β-Sheet',
  C: 'Coil/Loop',
};

const SecondaryStructurePanel: React.FC<SecondaryStructurePanelProps> = ({
  sequence,
  secondaryStructure,
  isLoading = false,
  source = 'sequence_estimate',
}) => {
  const theme = useTheme();

  // Generate simple SS estimate from sequence if not provided
  const estimatedSS = useMemo(() => {
    if (secondaryStructure?.assignments) {
      return secondaryStructure.assignments;
    }
    
    // Simple propensity-based estimate
    const HELIX = new Set('AELMQKR');
    const SHEET = new Set('VIYFTW');
    
    let ss = '';
    for (const aa of sequence.toUpperCase()) {
      if (HELIX.has(aa)) ss += 'H';
      else if (SHEET.has(aa)) ss += 'E';
      else ss += 'C';
    }
    return ss;
  }, [sequence, secondaryStructure?.assignments]);

  // Calculate percentages
  const stats = useMemo(() => {
    if (secondaryStructure) {
      return {
        helix: secondaryStructure.helix_percent ?? 0,
        sheet: secondaryStructure.sheet_percent ?? 0,
        coil: secondaryStructure.coil_percent ?? 0,
        helixCount: secondaryStructure.helix_count ?? 0,
        sheetCount: secondaryStructure.sheet_count ?? 0,
        coilCount: secondaryStructure.coil_count ?? 0,
      };
    }
    
    // Calculate from estimated SS
    const total = estimatedSS.length;
    const helixCount = (estimatedSS.match(/H/g) || []).length;
    const sheetCount = (estimatedSS.match(/E/g) || []).length;
    const coilCount = total - helixCount - sheetCount;
    
    return {
      helix: (helixCount / total) * 100,
      sheet: (sheetCount / total) * 100,
      coil: (coilCount / total) * 100,
      helixCount,
      sheetCount,
      coilCount,
    };
  }, [estimatedSS, secondaryStructure]);

  // Create visual representation of SS sequence
  const renderSSSequence = () => {
    const ss = estimatedSS;
    const chunkSize = 10;
    const rows = [];
    
    for (let i = 0; i < ss.length; i += chunkSize) {
      const chunk = ss.slice(i, i + chunkSize);
      const seqChunk = sequence.slice(i, i + chunkSize);
      
      rows.push(
        <Box key={i} sx={{ display: 'flex', mb: 0.5 }}>
          <Typography 
            variant="caption" 
            sx={{ 
              width: 40, 
              color: 'text.secondary',
              fontFamily: 'monospace',
            }}
          >
            {i + 1}
          </Typography>
          <Box sx={{ display: 'flex', gap: 0.25 }}>
            {chunk.split('').map((ssChar: string, j: number) => (
              <Tooltip 
                key={j} 
                title={`${i + j + 1}: ${seqChunk[j] || '?'} (${SS_LABELS[ssChar as keyof typeof SS_LABELS] || 'Unknown'})`}
                arrow
              >
                <Box
                  sx={{
                    width: 16,
                    height: 20,
                    backgroundColor: SS_COLORS[ssChar as keyof typeof SS_COLORS] || SS_COLORS.C,
                    borderRadius: 0.5,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '0.6rem',
                    fontFamily: 'monospace',
                    color: 'white',
                    fontWeight: 'bold',
                    cursor: 'pointer',
                    '&:hover': {
                      transform: 'scale(1.2)',
                      zIndex: 1,
                    },
                    transition: 'transform 0.1s',
                  }}
                >
                  {seqChunk[j] || ''}
                </Box>
              </Tooltip>
            ))}
          </Box>
        </Box>
      );
    }
    
    return rows;
  };

  return (
    <Paper 
      elevation={2} 
      sx={{ 
        p: 2, 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
        <Box display="flex" alignItems="center" gap={1}>
          <ScienceIcon sx={{ color: theme.palette.primary.main }} />
          <Typography variant="h6" fontWeight="bold">
            Secondary Structure
          </Typography>
        </Box>
        <Chip 
          label={source === 'structure' ? 'From Structure' : source === 'live' ? 'Live' : 'Estimated'}
          size="small"
          color={source === 'structure' ? 'success' : source === 'live' ? 'primary' : 'default'}
          variant="outlined"
        />
      </Box>

      {isLoading ? (
        <Box sx={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <LinearProgress sx={{ width: '80%' }} />
        </Box>
      ) : (
        <>
          {/* Statistics Cards */}
          <Stack direction="row" spacing={1} mb={2}>
            <Box 
              sx={{ 
                flex: 1, 
                p: 1.5, 
                borderRadius: 1,
                backgroundColor: alpha(SS_COLORS.H, 0.1),
                border: `1px solid ${alpha(SS_COLORS.H, 0.3)}`,
              }}
            >
              <Box display="flex" alignItems="center" gap={0.5} mb={0.5}>
                <HelixIcon sx={{ fontSize: 16, color: SS_COLORS.H }} />
                <Typography variant="caption" color="text.secondary">
                  α-Helix
                </Typography>
              </Box>
              <Typography variant="h6" sx={{ color: SS_COLORS.H, fontWeight: 'bold' }}>
                {stats.helix.toFixed(1)}%
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {stats.helixCount} residues
              </Typography>
            </Box>

            <Box 
              sx={{ 
                flex: 1, 
                p: 1.5, 
                borderRadius: 1,
                backgroundColor: alpha(SS_COLORS.E, 0.1),
                border: `1px solid ${alpha(SS_COLORS.E, 0.3)}`,
              }}
            >
              <Box display="flex" alignItems="center" gap={0.5} mb={0.5}>
                <SheetIcon sx={{ fontSize: 16, color: SS_COLORS.E }} />
                <Typography variant="caption" color="text.secondary">
                  β-Sheet
                </Typography>
              </Box>
              <Typography variant="h6" sx={{ color: SS_COLORS.E, fontWeight: 'bold' }}>
                {stats.sheet.toFixed(1)}%
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {stats.sheetCount} residues
              </Typography>
            </Box>

            <Box 
              sx={{ 
                flex: 1, 
                p: 1.5, 
                borderRadius: 1,
                backgroundColor: alpha(SS_COLORS.C, 0.1),
                border: `1px solid ${alpha(SS_COLORS.C, 0.3)}`,
              }}
            >
              <Box display="flex" alignItems="center" gap={0.5} mb={0.5}>
                <CoilIcon sx={{ fontSize: 16, color: SS_COLORS.C }} />
                <Typography variant="caption" color="text.secondary">
                  Coil/Loop
                </Typography>
              </Box>
              <Typography variant="h6" sx={{ color: SS_COLORS.C, fontWeight: 'bold' }}>
                {stats.coil.toFixed(1)}%
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {stats.coilCount} residues
              </Typography>
            </Box>
          </Stack>

          {/* Stacked Progress Bar */}
          <Box mb={2}>
            <Box 
              sx={{ 
                display: 'flex', 
                height: 12, 
                borderRadius: 6, 
                overflow: 'hidden',
                backgroundColor: alpha(theme.palette.divider, 0.2),
              }}
            >
              <Box 
                sx={{ 
                  width: `${stats.helix}%`, 
                  backgroundColor: SS_COLORS.H,
                  transition: 'width 0.3s ease',
                }} 
              />
              <Box 
                sx={{ 
                  width: `${stats.sheet}%`, 
                  backgroundColor: SS_COLORS.E,
                  transition: 'width 0.3s ease',
                }} 
              />
              <Box 
                sx={{ 
                  width: `${stats.coil}%`, 
                  backgroundColor: SS_COLORS.C,
                  transition: 'width 0.3s ease',
                }} 
              />
            </Box>
          </Box>

          <Divider sx={{ mb: 2 }} />

          {/* Sequence Visualization */}
          <Typography variant="subtitle2" color="text.secondary" mb={1}>
            Residue Map ({sequence.length} residues)
          </Typography>
          
          <Box 
            sx={{ 
              flex: 1, 
              overflow: 'auto',
              backgroundColor: alpha(theme.palette.background.default, 0.5),
              borderRadius: 1,
              p: 1,
            }}
          >
            {renderSSSequence()}
          </Box>

          {/* Legend */}
          <Box display="flex" gap={2} mt={2} justifyContent="center">
            {Object.entries(SS_LABELS).map(([key, label]) => (
              <Box key={key} display="flex" alignItems="center" gap={0.5}>
                <Box 
                  sx={{ 
                    width: 12, 
                    height: 12, 
                    borderRadius: '50%',
                    backgroundColor: SS_COLORS[key as keyof typeof SS_COLORS],
                  }} 
                />
                <Typography variant="caption">{label}</Typography>
              </Box>
            ))}
          </Box>
        </>
      )}
    </Paper>
  );
};

export default SecondaryStructurePanel;
