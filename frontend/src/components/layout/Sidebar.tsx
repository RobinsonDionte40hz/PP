import { useNavigate, useLocation } from 'react-router-dom';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
  Box,
  Toolbar,
} from '@mui/material';
import DashboardIcon from '@mui/icons-material/Dashboard';
import AddCircleIcon from '@mui/icons-material/AddCircle';
import MonitorHeartIcon from '@mui/icons-material/MonitorHeart';
import HistoryIcon from '@mui/icons-material/History';
import CampaignIcon from '@mui/icons-material/Campaign';
import ScienceIcon from '@mui/icons-material/Science';
import SettingsIcon from '@mui/icons-material/Settings';

interface SidebarProps {
  open: boolean;
  onClose: () => void;
  drawerWidth: number;
  isMobile: boolean;
}

interface NavItem {
  text: string;
  icon: React.ReactElement;
  path: string;
}

const mainNavItems: NavItem[] = [
  { text: 'Dashboard', icon: <DashboardIcon />, path: '/dashboard' },
  { text: 'New Prediction', icon: <AddCircleIcon />, path: '/dashboard/predict' },
  { text: 'Live Monitoring', icon: <MonitorHeartIcon />, path: '/dashboard/monitor/active' },
  { text: 'History', icon: <HistoryIcon />, path: '/dashboard/history' },
];

const secondaryNavItems: NavItem[] = [
  { text: 'Aggregation Screening', icon: <ScienceIcon />, path: '/dashboard/screening' },
  { text: 'Campaign Management', icon: <CampaignIcon />, path: '/dashboard/campaigns' },
];

const settingsNavItems: NavItem[] = [
  { text: 'Settings', icon: <SettingsIcon />, path: '/dashboard/settings' },
];

export default function Sidebar({ open, onClose, drawerWidth, isMobile }: SidebarProps) {
  const navigate = useNavigate();
  const location = useLocation();

  const handleNavigate = (path: string) => {
    navigate(path);
    if (isMobile) {
      onClose();
    }
  };

  const renderNavItems = (items: NavItem[]) => (
    <List>
      {items.map((item) => {
        const isActive = location.pathname === item.path || 
                        (item.path !== '/dashboard' && location.pathname.startsWith(item.path));
        
        return (
          <ListItem key={item.text} disablePadding>
            <ListItemButton
              onClick={() => handleNavigate(item.path)}
              selected={isActive}
              sx={{
                '&.Mui-selected': {
                  backgroundColor: 'primary.light',
                  '&:hover': {
                    backgroundColor: 'primary.light',
                  },
                },
              }}
            >
              <ListItemIcon sx={{ color: isActive ? 'primary.main' : 'inherit' }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItemButton>
          </ListItem>
        );
      })}
    </List>
  );

  const drawerContent = (
    <Box sx={{ overflow: 'auto' }}>
      <Toolbar /> {/* Spacer for AppBar */}
      
      {renderNavItems(mainNavItems)}
      
      <Divider sx={{ my: 1 }} />
      
      {renderNavItems(secondaryNavItems)}
      
      <Divider sx={{ my: 1 }} />
      
      {renderNavItems(settingsNavItems)}
    </Box>
  );

  return (
    <Drawer
      variant={isMobile ? 'temporary' : 'persistent'}
      open={open}
      onClose={onClose}
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: drawerWidth,
          boxSizing: 'border-box',
        },
      }}
    >
      {drawerContent}
    </Drawer>
  );
}
