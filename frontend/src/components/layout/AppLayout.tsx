import { useState } from 'react';
import { Outlet } from 'react-router-dom';
import { Box, useTheme, useMediaQuery } from '@mui/material';
import Header from './Header.tsx';
import Sidebar from './Sidebar.tsx';
import Footer from './Footer.tsx';
import { useKeyboardShortcuts } from '../../hooks/useKeyboardShortcuts';

const DRAWER_WIDTH = 240;

export default function AppLayout() {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [sidebarOpen, setSidebarOpen] = useState(!isMobile);

  // Enable global keyboard shortcuts
  useKeyboardShortcuts();

  const handleSidebarToggle = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', flexDirection: 'column' }}>
      <Header onMenuClick={handleSidebarToggle} />
      
      <Box sx={{ display: 'flex', flex: 1, pt: '64px' }}>
        <Sidebar 
          open={sidebarOpen} 
          onClose={() => setSidebarOpen(false)}
          drawerWidth={DRAWER_WIDTH}
          isMobile={isMobile}
        />
        
        <Box
          component="main"
          sx={{
            flexGrow: 1,
            p: { xs: 2, sm: 3 },
            width: { md: `calc(100% - ${sidebarOpen ? DRAWER_WIDTH : 0}px)` },
            ml: { md: sidebarOpen ? `${DRAWER_WIDTH}px` : 0 },
            transition: theme.transitions.create(['margin', 'width'], {
              easing: theme.transitions.easing.sharp,
              duration: theme.transitions.duration.leavingScreen,
            }),
            minHeight: 'calc(100vh - 64px - 64px)', // viewport - header - footer
          }}
        >
          <Outlet />
        </Box>
      </Box>
      
      <Footer />
    </Box>
  );
}
