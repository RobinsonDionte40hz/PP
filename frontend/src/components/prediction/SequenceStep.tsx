import React, { useState } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  Alert,
  alpha,
  useTheme,
  CircularProgress,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Science as ScienceIcon,
  CheckCircle,
} from '@mui/icons-material';

interface SequenceStepProps {
  formData: {
    sequence: string;
    native_pdb_id?: string;
  };
  onChange: (updates: Partial<SequenceStepProps['formData']>) => void;
}

const SequenceStep: React.FC<SequenceStepProps> = ({ formData, onChange }) => {
  const theme = useTheme();
  const [sequenceError, setSequenceError] = useState<string>('');
  const [pdbLoading, setPdbLoading] = useState(false);
  const [pdbError, setPdbError] = useState<string>('');

  const validateSequence = (seq: string): boolean => {
    // Remove whitespace and convert to uppercase
    const cleanSeq = seq.replace(/\s/g, '').toUpperCase();
    
    // Check if empty
    if (cleanSeq.length === 0) {
      setSequenceError('');
      return false;
    }

    // Check for valid amino acid codes
    const validAminoAcids = /^[ACDEFGHIKLMNPQRSTVWY]+$/;
    if (!validAminoAcids.test(cleanSeq)) {
      setSequenceError('Sequence contains invalid amino acid codes. Use single-letter codes (A-Z).');
      return false;
    }

    // Check minimum length
    if (cleanSeq.length < 3) {
      setSequenceError('Sequence must be at least 3 amino acids long.');
      return false;
    }

    setSequenceError('');
    return true;
  };

  const handleSequenceChange = (value: string) => {
    const cleanSeq = value.replace(/\s/g, '').toUpperCase();
    onChange({ sequence: cleanSeq });
    validateSequence(cleanSeq);
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      
      // Parse FASTA format
      if (content.startsWith('>')) {
        const lines = content.split('\n');
        const sequence = lines.slice(1).join('').replace(/\s/g, '').toUpperCase();
        handleSequenceChange(sequence);
      } else {
        handleSequenceChange(content);
      }
    };
    reader.readAsText(file);
  };

  const loadExampleSequence = () => {
    // 1UBQ - Ubiquitin (76 residues)
    const exampleSeq = 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG';
    handleSequenceChange(exampleSeq);
    onChange({ native_pdb_id: '1UBQ' });
  };

  const loadFromPDB = async () => {
    const pdbId = formData.native_pdb_id?.trim().toUpperCase();
    if (!pdbId || pdbId.length !== 4) {
      setPdbError('Please enter a valid 4-character PDB ID');
      return;
    }

    setPdbLoading(true);
    setPdbError('');

    try {
      // Fetch PDB file from RCSB
      const response = await fetch(`https://files.rcsb.org/download/${pdbId}.pdb`);
      
      if (!response.ok) {
        throw new Error(`PDB ${pdbId} not found`);
      }

      const pdbText = await response.text();
      
      // Extract amino acids from ATOM records (only structured/observed residues)
      const aaMap: { [key: string]: string } = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
      };

      // Parse ATOM records for CA atoms (one per residue) from chain A
      const lines = pdbText.split('\n');
      const residues: { resNum: number; resName: string }[] = [];
      const seenResidues = new Set<string>();
      
      for (const line of lines) {
        if (line.startsWith('ATOM') || line.startsWith('HETATM')) {
          const atomName = line.substring(12, 16).trim();
          const resName = line.substring(17, 20).trim();
          const chainId = line.substring(21, 22);
          const resSeq = parseInt(line.substring(22, 26).trim(), 10);
          
          // Only process CA atoms from first chain (usually 'A' or ' ')
          if (atomName === 'CA' && aaMap[resName]) {
            const key = `${chainId}:${resSeq}`;
            if (!seenResidues.has(key)) {
              seenResidues.add(key);
              residues.push({ resNum: resSeq, resName });
            }
          }
        }
        // Stop at first TER or ENDMDL (single chain/model only)
        if (line.startsWith('TER') || line.startsWith('ENDMDL')) {
          if (residues.length > 0) break;
        }
      }

      // Sort by residue number and build sequence
      residues.sort((a, b) => a.resNum - b.resNum);
      let sequence = residues.map(r => aaMap[r.resName]).join('');

      if (sequence.length === 0) {
        // Fallback to SEQRES if no ATOM records found
        const seqresLines = lines.filter(line => line.startsWith('SEQRES'));
        seqresLines.forEach(line => {
          const parts = line.split(/\s+/).slice(4);
          parts.forEach(aa => {
            if (aaMap[aa]) {
              sequence += aaMap[aa];
            }
          });
        });
      }

      if (sequence.length === 0) {
        throw new Error('Could not extract sequence from PDB file');
      }

      // Update form with sequence
      handleSequenceChange(sequence);
      setPdbError('');
      
    } catch (error) {
      setPdbError(error instanceof Error ? error.message : 'Failed to load PDB');
    } finally {
      setPdbLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h6" fontWeight="bold" gutterBottom>
        Protein Sequence
      </Typography>
      <Typography variant="body2" color="text.secondary" mb={3}>
        Enter the amino acid sequence in single-letter code format
      </Typography>

      {/* Sequence Input */}
      <TextField
        fullWidth
        multiline
        rows={6}
        value={formData.sequence}
        onChange={(e) => handleSequenceChange(e.target.value)}
        placeholder="Enter protein sequence (e.g., MQIFVKTLT...)"
        error={Boolean(sequenceError)}
        helperText={sequenceError || `${formData.sequence.length} amino acids`}
        sx={{ mb: 2 }}
      />

      {/* Action Buttons */}
      <Box display="flex" gap={2} mb={3}>
        <Button
          variant="outlined"
          startIcon={<UploadIcon />}
          component="label"
        >
          Upload FASTA
          <input
            type="file"
            accept=".fasta,.fa,.txt"
            hidden
            onChange={handleFileUpload}
          />
        </Button>
        <Button
          variant="outlined"
          startIcon={<ScienceIcon />}
          onClick={loadExampleSequence}
        >
          Load Example (1UBQ)
        </Button>
      </Box>

      {/* Native PDB ID (Optional) */}
      <Box
        sx={{
          p: 2,
          backgroundColor: alpha(theme.palette.info.main, 0.05),
          borderRadius: 1,
          border: `1px solid ${alpha(theme.palette.info.main, 0.2)}`,
        }}
      >
        <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
          Native Structure (Optional)
        </Typography>
        <Typography variant="caption" color="text.secondary" display="block" mb={2}>
          Provide a PDB ID to calculate RMSD and validate structure quality
        </Typography>
        <Box display="flex" gap={2} alignItems="flex-start">
          <TextField
            fullWidth
            size="small"
            value={formData.native_pdb_id || ''}
            onChange={(e) => {
              const value = e.target.value.toUpperCase().trim();
              onChange({ native_pdb_id: value });
              setPdbError('');
            }}
            placeholder="e.g., 1UBQ"
            error={Boolean(pdbError) || (formData.native_pdb_id ? formData.native_pdb_id.length !== 4 : false)}
            helperText={
              pdbError ||
              (formData.native_pdb_id 
                ? formData.native_pdb_id.length === 4 
                  ? `✓ PDB ID: ${formData.native_pdb_id} will be used for RMSD calculation`
                  : `PDB ID must be exactly 4 characters (currently ${formData.native_pdb_id.length})`
                : "4-character PDB identifier (e.g., 1UBQ, 2MR9, 1CRN)")
            }
            InputProps={{
              endAdornment: formData.native_pdb_id && formData.native_pdb_id.length === 4 && !pdbError ? (
                <CheckCircle color="success" fontSize="small" />
              ) : null,
            }}
          />
          <Button
            variant="outlined"
            onClick={loadFromPDB}
            disabled={!formData.native_pdb_id || formData.native_pdb_id.length !== 4 || pdbLoading}
            startIcon={pdbLoading ? <CircularProgress size={16} /> : <ScienceIcon />}
            sx={{ minWidth: 140, whiteSpace: 'nowrap' }}
          >
            {pdbLoading ? 'Loading...' : 'Load from PDB'}
          </Button>
        </Box>
      </Box>

      {/* Info Alert */}
      {formData.sequence.length > 0 && !sequenceError && (
        <Alert severity="success" sx={{ mt: 3 }}>
          Sequence validated: {formData.sequence.length} amino acids
          {formData.native_pdb_id && ` • Native structure: ${formData.native_pdb_id}`}
        </Alert>
      )}
    </Box>
  );
};

export default SequenceStep;
