import { useState } from 'react';
import { AppBar, Toolbar, IconButton, Typography, Box, Chip, Tooltip, Menu, MenuItem, Avatar, ListItemIcon, Divider, CircularProgress } from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import NotificationsIcon from '@mui/icons-material/Notifications';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import LogoutIcon from '@mui/icons-material/Logout';
import SettingsIcon from '@mui/icons-material/Settings';
import { useNavigate } from 'react-router-dom';
import { useThemeStore } from '../../store/themeStore.ts';
import { useAuth } from '../../hooks/useAuth';
import KeyboardShortcutsDialog from '../common/KeyboardShortcutsDialog';

interface HeaderProps {
  onMenuClick: () => void;
}

export default function Header({ onMenuClick }: HeaderProps) {
  const { mode, toggleTheme } = useThemeStore();
  const { user, isAuthenticated, logout, isLoading } = useAuth();
  const navigate = useNavigate();
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const userMenuOpen = Boolean(anchorEl);

  const handleUserMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleUserMenuClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = async () => {
    handleUserMenuClose();
    await logout();
  };

  const handleSettings = () => {
    handleUserMenuClose();
    navigate('/settings');
  };

  const getInitials = (username: string) => {
    return username.substring(0, 2).toUpperCase();
  };

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
        
        {/* User Menu - Only show if authenticated */}
        {isAuthenticated && user && (
          <>
            <Tooltip title="Account">
              <IconButton
                onClick={handleUserMenuOpen}
                size="small"
                sx={{ ml: 2 }}
                aria-controls={userMenuOpen ? 'account-menu' : undefined}
                aria-haspopup="true"
                aria-expanded={userMenuOpen ? 'true' : undefined}
              >
                <Avatar sx={{ width: 32, height: 32, bgcolor: 'primary.main' }}>
                  {getInitials(user.username)}
                </Avatar>
              </IconButton>
            </Tooltip>
            <Menu
              anchorEl={anchorEl}
              id="account-menu"
              open={userMenuOpen}
              onClose={handleUserMenuClose}
              onClick={handleUserMenuClose}
              transformOrigin={{ horizontal: 'right', vertical: 'top' }}
              anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
              PaperProps={{
                elevation: 3,
                sx: {
                  mt: 1.5,
                  minWidth: 200,
                  '& .MuiAvatar-root': {
                    width: 32,
                    height: 32,
                    ml: -0.5,
                    mr: 1,
                  },
                },
              }}
            >
              <MenuItem disabled>
                <ListItemIcon>
                  <AccountCircleIcon fontSize="small" />
                </ListItemIcon>
                <Box>
                  <Typography variant="body2" fontWeight="bold">
                    {user.username}
                  </Typography>
                  {user.email && (
                    <Typography variant="caption" color="text.secondary">
                      {user.email}
                    </Typography>
                  )}
                </Box>
              </MenuItem>
              <Divider />
              <MenuItem onClick={handleSettings}>
                <ListItemIcon>
                  <SettingsIcon fontSize="small" />
                </ListItemIcon>
                Settings
              </MenuItem>
              <Divider />
              <MenuItem onClick={handleLogout} disabled={isLoading}>
                <ListItemIcon>
                  {isLoading ? (
                    <CircularProgress size={20} />
                  ) : (
                    <LogoutIcon fontSize="small" />
                  )}
                </ListItemIcon>
                {isLoading ? 'Logging out...' : 'Logout'}
              </MenuItem>
            </Menu>
          </>
        )}
      </Toolbar>
      
      <KeyboardShortcutsDialog open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />
    </AppBar>
  );
}
