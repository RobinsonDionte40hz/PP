import { useState } from 'react';
import { AppBar, Toolbar, IconButton, Typography, Box, Chip, Tooltip } from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import NotificationsIcon from '@mui/icons-material/Notifications';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import { useThemeStore } from '../../store/themeStore.ts';
import KeyboardShortcutsDialog from '../common/KeyboardShortcutsDialog';

interface HeaderProps {
  onMenuClick: () => void;
}

export default function Header({ onMenuClick }: HeaderProps) {
  const { mode, toggleTheme } = useThemeStore();
  const [shortcutsOpen, setShortcutsOpen] = useState(false);

  return (
    <AppBar 
      position="fixed" 
      sx={{ 
        zIndex: (theme) => theme.zIndex.drawer + 1,
        backgroundColor: 'background.paper',
        color: 'text.primary',
        boxShadow: 1,
      }}
    >
      <Toolbar>
        <IconButton
          color="inherit"
          aria-label="open drawer"
          edge="start"
          onClick={onMenuClick}
          sx={{ mr: 2 }}
        >
          <MenuIcon />
        </IconButton>
        
        <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 0, mr: 2 }}>
          Protein Prediction Platform
        </Typography>
        
        <Chip 
          label="QCPP + UBF" 
          size="small" 
          color="primary" 
          variant="outlined"
          sx={{ ml: 1 }}
        />
        
        <Box sx={{ flexGrow: 1 }} />
        
        {/* System Status */}
        <Chip 
          label="System: Online" 
          size="small" 
          color="success" 
          sx={{ mr: 2 }}
        />
        
        {/* Keyboard Shortcuts Help */}
        <Tooltip title="Keyboard shortcuts (Ctrl+/)">
          <IconButton color="inherit" sx={{ mr: 1 }} onClick={() => setShortcutsOpen(true)}>
            <HelpOutlineIcon />
          </IconButton>
        </Tooltip>
        
        {/* Notifications */}
        <Tooltip title="Notifications">
          <IconButton color="inherit" sx={{ mr: 1 }}>
            <NotificationsIcon />
          </IconButton>
        </Tooltip>
        
        {/* Theme Toggle */}
        <Tooltip title={`Switch to ${mode === 'light' ? 'dark' : 'light'} mode`}>
          <IconButton onClick={toggleTheme} color="inherit">
            {mode === 'dark' ? <Brightness7Icon /> : <Brightness4Icon />}
          </IconButton>
        </Tooltip>
      </Toolbar>
      
      <KeyboardShortcutsDialog open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />
    </AppBar>
  );
}
