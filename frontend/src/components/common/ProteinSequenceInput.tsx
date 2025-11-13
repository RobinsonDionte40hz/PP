import { TextField, Box, Typography, Chip } from '@mui/material';
import { useState } from 'react';
import type { ChangeEvent } from 'react';

interface ProteinSequenceInputProps {
  value: string;
  onChange: (value: string) => void;
  error?: string;
  maxLength?: number;
  label?: string;
  helperText?: string;
}

export default function ProteinSequenceInput({
  value,
  onChange,
  error,
  maxLength = 1000,
  label = 'Protein Sequence',
  helperText = 'Enter amino acid sequence (single letter codes)',
}: ProteinSequenceInputProps) {
  const [validationError, setValidationError] = useState<string>('');

  const handleChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = event.target.value.toUpperCase();
    
    // Validate sequence (only valid amino acid codes)
    const validAminoAcids = /^[ACDEFGHIKLMNPQRSTVWY]*$/;
    if (!validAminoAcids.test(newValue)) {
      setValidationError('Invalid amino acid code. Use only: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y');
      return;
    }
    
    setValidationError('');
    onChange(newValue);
  };

  const sequenceLength = value.length;
  const isValid = !error && !validationError && sequenceLength > 0;

  return (
    <Box>
      <TextField
        fullWidth
        multiline
        rows={4}
        label={label}
        value={value}
        onChange={handleChange}
        error={Boolean(error || validationError)}
        helperText={validationError || error || helperText}
        inputProps={{
          maxLength,
          style: { fontFamily: 'monospace', fontSize: '14px' },
        }}
      />
      
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1, alignItems: 'center' }}>
        <Typography variant="caption" color="text.secondary">
          Length: {sequenceLength} / {maxLength}
        </Typography>
        
        {isValid && (
          <Chip
            label="Valid Sequence"
            color="success"
            size="small"
            variant="outlined"
          />
        )}
      </Box>
    </Box>
  );
}
