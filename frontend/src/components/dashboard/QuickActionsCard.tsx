import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Button,
  Box,
  useTheme,
  alpha,
} from '@mui/material';
import {
  Add as AddIcon,
  PlayArrow as PlayIcon,
  History as HistoryIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';
import type { NavigateFunction } from 'react-router-dom';
import { animationStyles } from '../../utils/animations';

interface QuickActionsCardProps {
  onNavigate: NavigateFunction;
}

interface ActionButton {
  label: string;
  icon: React.ReactNode;
  color: 'primary' | 'secondary' | 'success' | 'info';
  path: string;
  description: string;
}

const QuickActionsCard: React.FC<QuickActionsCardProps> = ({ onNavigate }) => {
  const theme = useTheme();

  const actions: ActionButton[] = [
    {
      label: 'New Prediction',
      icon: <AddIcon />,
      color: 'primary',
      path: '/dashboard/predictions/new',
      description: 'Start a new protein structure prediction',
    },
    {
      label: 'View Running',
      icon: <PlayIcon />,
      color: 'success',
      path: '/dashboard/monitor/active',
      description: 'Monitor active predictions',
    },
    {
      label: 'Browse History',
      icon: <HistoryIcon />,
      color: 'info',
      path: '/dashboard/history',
      description: 'View past predictions and results',
    },
    {
      label: 'Settings',
      icon: <SettingsIcon />,
      color: 'secondary',
      path: '/dashboard/settings',
      description: 'Configure system settings',
    },
  ];

  return (
    <Card
      elevation={2}
      sx={{
        height: '100%',
        background: `linear-gradient(135deg, ${alpha(
          theme.palette.primary.main,
          0.05
        )} 0%, ${alpha(theme.palette.background.paper, 1)} 100%)`,
      }}
    >
      <CardContent>
        <Typography variant="h6" fontWeight="bold" gutterBottom>
          Quick Actions
        </Typography>
        <Typography variant="body2" color="text.secondary" mb={3}>
          Get started with common tasks
        </Typography>

        <Box display="flex" flexWrap="wrap" gap={2}>
          {actions.map((action, index) => (
            <Box 
              key={action.label} 
              flex="1 1 calc(50% - 8px)" 
              minWidth="250px"
              sx={animationStyles.staggerItem(index)}
            >
              <Button
                variant="outlined"
                color={action.color}
                fullWidth
                startIcon={action.icon}
                onClick={() => onNavigate(action.path)}
                sx={{
                  py: 2,
                  justifyContent: 'flex-start',
                  textAlign: 'left',
                  flexDirection: 'column',
                  alignItems: 'flex-start',
                  height: '100%',
                  minHeight: 100,
                  '&:hover': {
                    backgroundColor: alpha(theme.palette[action.color].main, 0.1),
                    transform: 'translateY(-2px)',
                    boxShadow: 2,
                  },
                  transition: 'all 0.2s ease-in-out',
                }}
              >
                <Box display="flex" alignItems="center" mb={1}>
                  {action.icon}
                  <Typography
                    variant="body1"
                    fontWeight="bold"
                    ml={1}
                    sx={{ textTransform: 'none' }}
                  >
                    {action.label}
                  </Typography>
                </Box>
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ textTransform: 'none' }}
                >
                  {action.description}
                </Typography>
              </Button>
            </Box>
          ))}
        </Box>
      </CardContent>
    </Card>
  );
};

export default QuickActionsCard;
