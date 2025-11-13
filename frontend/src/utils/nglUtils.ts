/**
 * NGL Viewer Utilities
 * Helper functions for protein visualization using NGL Viewer
 */

import * as NGL from 'ngl';

export interface ProteinRepresentation {
  type: 'cartoon' | 'ball+stick' | 'licorice' | 'ribbon' | 'rope' | 'surface' | 'trace' | 'backbone';
  color: 'chainid' | 'residueindex' | 'atomindex' | 'bfactor' | 'hydrophobicity' | 'secondary structure' | 'random';
  params?: Record<string, string | number | boolean>;
}

export interface ViewerOptions {
  backgroundColor?: string;
  clipNear?: number;
  clipFar?: number;
  fogNear?: number;
  fogFar?: number;
  cameraType?: 'perspective' | 'orthographic';
  quality?: 'auto' | 'low' | 'medium' | 'high';
}

export const DEFAULT_REPRESENTATIONS: Record<string, ProteinRepresentation> = {
  cartoon: {
    type: 'cartoon',
    color: 'secondary structure',
    params: { aspectRatio: 5, smoothSheet: true }
  },
  ballStick: {
    type: 'ball+stick',
    color: 'chainid',
    params: { cylinderOnly: false }
  },
  surface: {
    type: 'surface',
    color: 'hydrophobicity',
    params: { surfaceType: 'ms', probeRadius: 1.4, smooth: 2 }
  },
  backbone: {
    type: 'backbone',
    color: 'residueindex',
    params: { linewidth: 8 }
  },
  licorice: {
    type: 'licorice',
    color: 'chainid',
    params: { multipleBond: 'symmetric' }
  }
};

export const COLOR_SCHEMES = {
  'secondary structure': 'Secondary Structure',
  'chainid': 'Chain ID',
  'residueindex': 'Residue Index',
  'hydrophobicity': 'Hydrophobicity',
  'bfactor': 'B-Factor',
  'atomindex': 'Atom Index',
  'random': 'Random'
};

/**
 * Initialize NGL stage with default settings
 */
export function createStage(
  container: HTMLElement,
  options: ViewerOptions = {}
): NGL.Stage {
  const defaultOptions: ViewerOptions = {
    backgroundColor: 'white',
    quality: 'high',
    cameraType: 'perspective',
    clipNear: 0,
    clipFar: 100,
    fogNear: 50,
    fogFar: 100,
    ...options
  };

  const stage = new NGL.Stage(container, defaultOptions);
  
  // Set up default lighting
  stage.setParameters({
    ambientIntensity: 0.5,
    lightIntensity: 1.0
  });

  return stage;
}

/**
 * Load PDB structure from string or file
 */
export async function loadStructure(
  stage: NGL.Stage,
  pdbData: string | File,
  name: string = 'structure'
): Promise<NGL.StructureComponent> {
  let structureComponent;

  if (typeof pdbData === 'string') {
    // Load from PDB string
    const blob = new Blob([pdbData], { type: 'text/plain' });
    structureComponent = await stage.loadFile(blob, { ext: 'pdb', name });
  } else {
    // Load from file
    structureComponent = await stage.loadFile(pdbData, { ext: 'pdb', name });
  }

  return structureComponent as NGL.StructureComponent;
}

/**
 * Add representation to structure
 */
export function addRepresentation(
  component: NGL.StructureComponent,
  representation: ProteinRepresentation
) {
  return component.addRepresentation(
    representation.type,
    {
      color: representation.color,
      ...representation.params
    }
  );
}

/**
 * Center and zoom structure
 */
export function centerStructure(
  stage: NGL.Stage,
  component?: NGL.StructureComponent
): void {
  if (component) {
    stage.autoView();
  } else {
    stage.autoView();
  }
}

/**
 * Highlight specific residues
 */
export function highlightResidues(
  component: NGL.StructureComponent,
  residueIndices: number[],
  color: string = '#ff0000'
) {
  if (residueIndices.length === 0) return null;

  const selection = residueIndices.map(i => `${i}`).join(' or ');
  return component.addRepresentation('ball+stick', {
    sele: selection,
    color: color,
    scale: 1.5
  });
}

/**
 * Compare two structures (predicted vs native)
 */
export async function loadComparisonStructures(
  stage: NGL.Stage,
  predictedPdb: string,
  nativePdb: string
): Promise<{
  predicted: NGL.StructureComponent;
  native: NGL.StructureComponent;
}> {
  const predicted = await loadStructure(stage, predictedPdb, 'predicted');
  const native = await loadStructure(stage, nativePdb, 'native');

  // Style predicted structure
  predicted.addRepresentation('cartoon', {
    color: '#4CAF50',
    opacity: 0.8
  });

  // Style native structure
  native.addRepresentation('cartoon', {
    color: '#2196F3',
    opacity: 0.5
  });

  centerStructure(stage);

  return { predicted, native };
}

/**
 * Highlight geometric patterns (phi/golden ratio)
 */
export function highlightGeometricPatterns(
  component: NGL.StructureComponent,
  patternType: 'phi' | 'icosahedron' | 'dodecahedron' | 'octahedron',
  residueIndices: number[]
) {
  if (residueIndices.length === 0) return null;

  const colors: Record<string, string> = {
    phi: '#FFD700',
    icosahedron: '#FF6B6B',
    dodecahedron: '#4ECDC4',
    octahedron: '#95E1D3'
  };

  const selection = residueIndices.map(i => `${i}`).join(' or ');
  return component.addRepresentation('spacefill', {
    sele: selection,
    color: colors[patternType] || '#FFD700',
    scale: 1.2
  });
}

/**
 * Create screenshot of current view
 */
export function takeScreenshot(
  stage: NGL.Stage,
  options: {
    factor?: number;
    antialias?: boolean;
    trim?: boolean;
    transparent?: boolean;
  } = {}
): Promise<Blob> {
  const defaultOptions = {
    factor: 2,
    antialias: true,
    trim: false,
    transparent: false,
    ...options
  };

  return new Promise((resolve) => {
    stage.makeImage(defaultOptions).then((blob) => {
      resolve(blob);
    });
  });
}

/**
 * Export structure as PDB file
 */
export function exportPDB(
  component: NGL.StructureComponent,
  filename: string = 'structure.pdb'
): void {
  const writer = new NGL.PdbWriter(component.structure);
  const pdbData = writer.getData();
  
  const blob = new Blob([pdbData], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  
  URL.revokeObjectURL(url);
}

/**
 * Get structure statistics
 */
export function getStructureStats(component: NGL.StructureComponent): {
  residueCount: number;
  atomCount: number;
  chainCount: number;
  bounds: { min: number[]; max: number[]; center: number[] };
} {
  const structure = component.structure;
  
  return {
    residueCount: structure.residueStore.count,
    atomCount: structure.atomStore.count,
    chainCount: structure.chainStore.count,
    bounds: {
      min: structure.boundingBox.min.toArray(),
      max: structure.boundingBox.max.toArray(),
      center: structure.center.toArray()
    }
  };
}

/**
 * Set quality level
 */
export function setQuality(
  stage: NGL.Stage,
  quality: 'low' | 'medium' | 'high' | 'auto'
): void {
  stage.setParameters({ quality });
}

/**
 * Dispose of NGL resources
 */
export function disposeStage(stage: NGL.Stage): void {
  stage.dispose();
}
