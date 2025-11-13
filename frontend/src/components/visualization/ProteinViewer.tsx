/**
 * ProteinViewer Component
 * 3D protein structure visualization using NGL Viewer
 */

import React, { useEffect, useRef, useState } from 'react';
import { Box, Paper, CircularProgress, Alert } from '@mui/material';
import * as NGL from 'ngl';
import {
  createStage,
  loadStructure,
  addRepresentation,
  centerStructure,
  highlightResidues,
  highlightGeometricPatterns,
  loadComparisonStructures,
  disposeStage,
  getStructureStats,
  DEFAULT_REPRESENTATIONS
} from '../../utils/nglUtils';
import type { ProteinRepresentation, ViewerOptions } from '../../utils/nglUtils';

export interface ProteinViewerProps {
  pdbData?: string;
  pdbFile?: File;
  nativePdbData?: string; // For comparison mode
  height?: number | string;
  width?: number | string;
  representation?: keyof typeof DEFAULT_REPRESENTATIONS;
  customRepresentation?: ProteinRepresentation;
  viewerOptions?: ViewerOptions;
  highlightedResidues?: number[];
  geometricPatterns?: {
    type: 'phi' | 'icosahedron' | 'dodecahedron' | 'octahedron';
    residues: number[];
  }[];
  onLoad?: (stats: ReturnType<typeof getStructureStats>) => void;
  onError?: (error: Error) => void;
}

export const ProteinViewer: React.FC<ProteinViewerProps> = ({
  pdbData,
  pdbFile,
  nativePdbData,
  height = 600,
  width = '100%',
  representation = 'cartoon',
  customRepresentation,
  viewerOptions,
  highlightedResidues = [],
  geometricPatterns = [],
  onLoad,
  onError
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<NGL.Stage | null>(null);
  const componentRef = useRef<NGL.StructureComponent | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    let mounted = true;

    const initViewer = async () => {
      try {
        setLoading(true);
        setError(null);

        // Create stage
        const stage = createStage(containerRef.current!, {
          backgroundColor: 'white',
          ...viewerOptions
        });
        stageRef.current = stage;

        // Handle window resize
        const handleResize = () => {
          stage.handleResize();
        };
        window.addEventListener('resize', handleResize);

        // Load structure(s)
        if (nativePdbData && pdbData) {
          // Comparison mode
          const { predicted } = await loadComparisonStructures(
            stage,
            pdbData,
            nativePdbData
          );
          componentRef.current = predicted;

          if (mounted && onLoad) {
            onLoad(getStructureStats(predicted));
          }
        } else if (pdbData || pdbFile) {
          // Single structure mode
          const component = await loadStructure(
            stage,
            pdbData || pdbFile!,
            'protein'
          );
          componentRef.current = component;

          // Add representation
          const rep = customRepresentation || DEFAULT_REPRESENTATIONS[representation];
          addRepresentation(component, rep);

          // Highlight residues if specified
          if (highlightedResidues.length > 0) {
            highlightResidues(component, highlightedResidues);
          }

          // Highlight geometric patterns
          geometricPatterns.forEach(pattern => {
            highlightGeometricPatterns(component, pattern.type, pattern.residues);
          });

          // Center view
          centerStructure(stage, component);

          if (mounted && onLoad) {
            onLoad(getStructureStats(component));
          }
        }

        if (mounted) {
          setLoading(false);
        }

        return () => {
          window.removeEventListener('resize', handleResize);
        };
      } catch (err) {
        if (mounted) {
          const errorMessage = err instanceof Error ? err.message : 'Failed to load structure';
          setError(errorMessage);
          setLoading(false);
          onError?.(err instanceof Error ? err : new Error(errorMessage));
        }
      }
    };

    initViewer();

    return () => {
      mounted = false;
      if (stageRef.current) {
        disposeStage(stageRef.current);
        stageRef.current = null;
      }
    };
  }, [pdbData, pdbFile, nativePdbData, representation, customRepresentation, viewerOptions, highlightedResidues, geometricPatterns, onLoad, onError]);

  // Update highlights when they change
  useEffect(() => {
    if (!componentRef.current) return;

    // Remove old highlights and add new ones
    if (highlightedResidues.length > 0) {
      highlightResidues(componentRef.current, highlightedResidues);
    }
  }, [highlightedResidues]);

  // Update geometric patterns when they change
  useEffect(() => {
    if (!componentRef.current) return;

    geometricPatterns.forEach(pattern => {
      highlightGeometricPatterns(componentRef.current!, pattern.type, pattern.residues);
    });
  }, [geometricPatterns]);

  return (
    <Paper sx={{ position: 'relative', height, width, overflow: 'hidden' }}>
      {loading && (
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: 'rgba(255, 255, 255, 0.9)',
            zIndex: 1
          }}
        >
          <CircularProgress />
        </Box>
      )}

      {error && (
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            p: 2,
            zIndex: 1
          }}
        >
          <Alert severity="error" sx={{ maxWidth: 500 }}>
            {error}
          </Alert>
        </Box>
      )}

      <Box
        ref={containerRef}
        sx={{
          width: '100%',
          height: '100%',
          '& canvas': {
            outline: 'none'
          }
        }}
      />
    </Paper>
  );
};

export default ProteinViewer;
