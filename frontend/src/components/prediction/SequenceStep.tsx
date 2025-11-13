import React, { useState } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  Alert,
  alpha,
  useTheme,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Science as ScienceIcon,
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
        <TextField
          fullWidth
          size="small"
          value={formData.native_pdb_id || ''}
          onChange={(e) => onChange({ native_pdb_id: e.target.value })}
          placeholder="e.g., 1UBQ"
          helperText="4-character PDB identifier (e.g., 1UBQ, 2MR9)"
        />
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
