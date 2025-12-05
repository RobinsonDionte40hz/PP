import React, { useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  LinearProgress,
  Tooltip,
  useTheme,
  alpha,
  Stack,
} from '@mui/material';
import {
  TrendingUp as HelixIcon,
  ViewStream as SheetIcon,
  Waves as CoilIcon,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';

interface SecondaryStructureSummaryProps {
  predictionId: string;
  sequence: string;
}

// Colors for secondary structure
const SS_COLORS = {
  H: '#e91e63', // Pink/magenta for helix
  E: '#2196f3', // Blue for sheet
  C: '#9e9e9e', // Gray for coil
};

const SecondaryStructureSummary: React.FC<SecondaryStructureSummaryProps> = ({
  predictionId,
  sequence,
}) => {
  const theme = useTheme();

  // Fetch secondary structure from API
  const { data: ssData, isLoading } = useQuery({
    queryKey: ['secondary-structure', predictionId],
    queryFn: async () => {
      const response = await fetch(`/api/results/${predictionId}/secondary-structure`);
      if (!response.ok) {
        throw new Error('Failed to fetch secondary structure');
      }
      return response.json();
    },
    staleTime: 60000, // Cache for 1 minute
  });

  const stats = useMemo(() => {
    if (ssData?.secondary_structure) {
      const ss = ssData.secondary_structure;
      return {
        helix: ss.helix_percent ?? 0,
        sheet: ss.sheet_percent ?? 0,
        coil: ss.coil_percent ?? 0,
        helixCount: ss.helix_count ?? 0,
        sheetCount: ss.sheet_count ?? 0,
        coilCount: ss.coil_count ?? 0,
        assignments: ss.assignments || '',
        source: ssData.source || 'unknown',
      };
    }
    
    // Estimate from sequence if no data
    const HELIX = new Set('AELMQKR');
    const SHEET = new Set('VIYFTW');
    
    let h = 0, e = 0;
    for (const aa of sequence.toUpperCase()) {
      if (HELIX.has(aa)) h++;
      else if (SHEET.has(aa)) e++;
    }
    const total = sequence.length;
    const c = total - h - e;
    
    return {
      helix: (h / total) * 100,
      sheet: (e / total) * 100,
      coil: (c / total) * 100,
      helixCount: h,
      sheetCount: e,
      coilCount: c,
      assignments: '',
      source: 'sequence_estimate',
    };
  }, [ssData, sequence]);

  // Create mini sequence visualization (condensed)
  const renderMiniSequence = () => {
    const ss = stats.assignments || '';
    if (!ss) return null;
    
    // Group into blocks
    const blockSize = Math.max(1, Math.ceil(ss.length / 40));
    const blocks = [];
    
    for (let i = 0; i < ss.length; i += blockSize) {
      const chunk = ss.slice(i, i + blockSize);
      // Find dominant SS in this block
      const h = (chunk.match(/H/g) || []).length;
      const e = (chunk.match(/E/g) || []).length;
      const c = chunk.length - h - e;
      
      let dominant = 'C';
      if (h > e && h > c) dominant = 'H';
      else if (e > h && e > c) dominant = 'E';
      
      blocks.push(
        <Tooltip 
          key={i} 
          title={`Residues ${i + 1}-${Math.min(i + blockSize, ss.length)}`}
          arrow
        >
          <Box
            sx={{
              width: 8,
              height: 16,
              backgroundColor: SS_COLORS[dominant as keyof typeof SS_COLORS],
              borderRadius: 0.5,
              cursor: 'pointer',
            }}
          />
        </Tooltip>
      );
    }
    
    return (
      <Box sx={{ display: 'flex', gap: 0.25, flexWrap: 'wrap', mt: 1 }}>
        {blocks}
      </Box>
    );
  };

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
        Secondary Structure
      </Typography>
      
      {isLoading ? (
        <LinearProgress />
      ) : (
        <>
          {/* Stacked Bar */}
          <Box 
            sx={{ 
              display: 'flex', 
              height: 8, 
              borderRadius: 4, 
              overflow: 'hidden',
              mb: 2,
              backgroundColor: alpha(theme.palette.divider, 0.2),
            }}
          >
            <Tooltip title={`α-Helix: ${stats.helix.toFixed(1)}%`}>
              <Box 
                sx={{ 
                  width: `${stats.helix}%`, 
                  backgroundColor: SS_COLORS.H,
                  transition: 'width 0.3s ease',
                }} 
              />
            </Tooltip>
            <Tooltip title={`β-Sheet: ${stats.sheet.toFixed(1)}%`}>
              <Box 
                sx={{ 
                  width: `${stats.sheet}%`, 
                  backgroundColor: SS_COLORS.E,
                  transition: 'width 0.3s ease',
                }} 
              />
            </Tooltip>
            <Tooltip title={`Coil/Loop: ${stats.coil.toFixed(1)}%`}>
              <Box 
                sx={{ 
                  width: `${stats.coil}%`, 
                  backgroundColor: SS_COLORS.C,
                  transition: 'width 0.3s ease',
                }} 
              />
            </Tooltip>
          </Box>

          {/* Stats Row */}
          <Stack direction="row" spacing={2} justifyContent="space-between">
            <Box display="flex" alignItems="center" gap={0.5}>
              <HelixIcon sx={{ fontSize: 16, color: SS_COLORS.H }} />
              <Typography variant="body2" fontWeight="medium" sx={{ color: SS_COLORS.H }}>
                {stats.helix.toFixed(1)}%
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Helix
              </Typography>
            </Box>
            
            <Box display="flex" alignItems="center" gap={0.5}>
              <SheetIcon sx={{ fontSize: 16, color: SS_COLORS.E }} />
              <Typography variant="body2" fontWeight="medium" sx={{ color: SS_COLORS.E }}>
                {stats.sheet.toFixed(1)}%
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Sheet
              </Typography>
            </Box>
            
            <Box display="flex" alignItems="center" gap={0.5}>
              <CoilIcon sx={{ fontSize: 16, color: SS_COLORS.C }} />
              <Typography variant="body2" fontWeight="medium" sx={{ color: SS_COLORS.C }}>
                {stats.coil.toFixed(1)}%
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Coil
              </Typography>
            </Box>
          </Stack>

          {/* Mini sequence visualization */}
          {renderMiniSequence()}

          {/* Source indicator */}
          <Typography 
            variant="caption" 
            color="text.secondary" 
            sx={{ display: 'block', mt: 1, textAlign: 'right' }}
          >
            {stats.source === 'structure' ? 'Calculated from structure' : 
             stats.source === 'sequence_estimate' ? 'Estimated from sequence' : 
             'From prediction'}
          </Typography>
        </>
      )}
    </Paper>
  );
};

export default SecondaryStructureSummary;
