import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  IconButton,
  Typography,
  Box,
  Chip,
  Divider,
} from '@mui/material';
import { Close as CloseIcon } from '@mui/icons-material';
import { KEYBOARD_SHORTCUTS } from '../../hooks/useKeyboardShortcuts';

interface KeyboardShortcutsDialogProps {
  open: boolean;
  onClose: () => void;
}

const KeyboardShortcutsDialog: React.FC<KeyboardShortcutsDialogProps> = ({ open, onClose }) => {
  // Detect if user is on Mac
  const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Typography variant="h6">Keyboard Shortcuts</Typography>
          <IconButton onClick={onClose} size="small">
            <CloseIcon />
          </IconButton>
        </Box>
      </DialogTitle>
      <DialogContent dividers>
        {KEYBOARD_SHORTCUTS.map((category, index) => (
          <Box key={category.category} mb={3}>
            <Typography variant="subtitle2" color="text.secondary" mb={2}>
              {category.category}
            </Typography>
            
            {category.shortcuts.map((shortcut) => (
              <Box
                key={shortcut.description}
                display="flex"
                justifyContent="space-between"
                alignItems="center"
                py={1}
              >
                <Typography variant="body2">{shortcut.description}</Typography>
                <Box display="flex" gap={0.5}>
                  {(isMac ? shortcut.mac : shortcut.keys).map((key) => (
                    <Chip
                      key={key}
                      label={key}
                      size="small"
                      sx={{
                        fontFamily: 'monospace',
                        fontSize: '0.75rem',
                        height: 24,
                        minWidth: 32,
                      }}
                    />
                  ))}
                </Box>
              </Box>
            ))}
            
            {index < KEYBOARD_SHORTCUTS.length - 1 && <Divider sx={{ mt: 2 }} />}
          </Box>
        ))}
        
        <Box mt={3} p={2} bgcolor="action.hover" borderRadius={1}>
          <Typography variant="caption" color="text.secondary">
            Tip: Press <Chip label="?" size="small" /> or <Chip label="Ctrl" size="small" /> + <Chip label="/" size="small" /> to open this dialog from anywhere in the app.
          </Typography>
        </Box>
      </DialogContent>
    </Dialog>
  );
};

export default KeyboardShortcutsDialog;
