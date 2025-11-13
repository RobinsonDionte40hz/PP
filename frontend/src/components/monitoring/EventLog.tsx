import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  InputAdornment,
  Chip,
  List,
  ListItem,
  ListItemText,
  IconButton,
  useTheme,
  alpha,
} from '@mui/material';
import {
  Search as SearchIcon,
  Clear as ClearIcon,
  Info as InfoIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  CheckCircle as SuccessIcon,
} from '@mui/icons-material';

interface EventLogProps {
  events: Array<{
    level: 'info' | 'warning' | 'error' | 'success';
    message: string;
    timestamp: string;
  }>;
}

const EventLog: React.FC<EventLogProps> = ({ events }) => {
  const theme = useTheme();
  const [filter, setFilter] = useState('');
  const [levelFilter, setLevelFilter] = useState<string | null>(null);
  const listEndRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  useEffect(() => {
    if (autoScroll && listEndRef.current) {
      listEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [events, autoScroll]);

  const filteredEvents = events.filter((event) => {
    const matchesText = event.message.toLowerCase().includes(filter.toLowerCase());
    const matchesLevel = !levelFilter || event.level === levelFilter;
    return matchesText && matchesLevel;
  });

  const getLevelIcon = (level: string) => {
    switch (level) {
      case 'info':
        return <InfoIcon sx={{ color: theme.palette.info.main, fontSize: 18 }} />;
      case 'warning':
        return <WarningIcon sx={{ color: theme.palette.warning.main, fontSize: 18 }} />;
      case 'error':
        return <ErrorIcon sx={{ color: theme.palette.error.main, fontSize: 18 }} />;
      case 'success':
        return <SuccessIcon sx={{ color: theme.palette.success.main, fontSize: 18 }} />;
      default:
        return <InfoIcon sx={{ color: theme.palette.info.main, fontSize: 18 }} />;
    }
  };

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'info':
        return theme.palette.info.main;
      case 'warning':
        return theme.palette.warning.main;
      case 'error':
        return theme.palette.error.main;
      case 'success':
        return theme.palette.success.main;
      default:
        return theme.palette.text.secondary;
    }
  };

  const levelCounts = {
    info: events.filter((e) => e.level === 'info').length,
    warning: events.filter((e) => e.level === 'warning').length,
    error: events.filter((e) => e.level === 'error').length,
    success: events.filter((e) => e.level === 'success').length,
  };

  return (
    <Paper elevation={2} sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Typography variant="h6" fontWeight="bold" mb={2}>
        Event Log
      </Typography>

      {/* Filter Chips */}
      <Box display="flex" gap={1} mb={2} flexWrap="wrap">
        <Chip
          label={`All (${events.length})`}
          onClick={() => setLevelFilter(null)}
          color={levelFilter === null ? 'primary' : 'default'}
          size="small"
        />
        <Chip
          label={`Info (${levelCounts.info})`}
          onClick={() => setLevelFilter('info')}
          color={levelFilter === 'info' ? 'info' : 'default'}
          size="small"
        />
        <Chip
          label={`Success (${levelCounts.success})`}
          onClick={() => setLevelFilter('success')}
          color={levelFilter === 'success' ? 'success' : 'default'}
          size="small"
        />
        <Chip
          label={`Warning (${levelCounts.warning})`}
          onClick={() => setLevelFilter('warning')}
          color={levelFilter === 'warning' ? 'warning' : 'default'}
          size="small"
        />
        <Chip
          label={`Error (${levelCounts.error})`}
          onClick={() => setLevelFilter('error')}
          color={levelFilter === 'error' ? 'error' : 'default'}
          size="small"
        />
      </Box>

      {/* Search Field */}
      <TextField
        fullWidth
        size="small"
        placeholder="Search events..."
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <SearchIcon fontSize="small" />
            </InputAdornment>
          ),
          endAdornment: filter && (
            <InputAdornment position="end">
              <IconButton size="small" onClick={() => setFilter('')}>
                <ClearIcon fontSize="small" />
              </IconButton>
            </InputAdornment>
          ),
        }}
        sx={{ mb: 2 }}
      />

      {/* Event List */}
      <Box
        sx={{
          flex: 1,
          overflow: 'auto',
          backgroundColor: alpha(theme.palette.background.default, 0.5),
          borderRadius: 1,
          p: 1,
        }}
        onScroll={(e) => {
          const { scrollTop, scrollHeight, clientHeight } = e.currentTarget;
          const isAtBottom = scrollHeight - scrollTop - clientHeight < 50;
          setAutoScroll(isAtBottom);
        }}
      >
        {filteredEvents.length === 0 ? (
          <Box
            display="flex"
            justifyContent="center"
            alignItems="center"
            height="100%"
            minHeight={200}
          >
            <Typography variant="body2" color="text.secondary">
              {events.length === 0 ? 'No events yet' : 'No matching events'}
            </Typography>
          </Box>
        ) : (
          <List dense sx={{ py: 0 }}>
            {filteredEvents.map((event, index) => (
              <ListItem
                key={index}
                sx={{
                  borderLeft: `3px solid ${getLevelColor(event.level)}`,
                  backgroundColor: alpha(getLevelColor(event.level), 0.05),
                  borderRadius: 0.5,
                  mb: 0.5,
                  py: 0.5,
                }}
              >
                <Box mr={1} display="flex" alignItems="center">
                  {getLevelIcon(event.level)}
                </Box>
                <ListItemText
                  primary={event.message}
                  secondary={new Date(event.timestamp).toLocaleTimeString()}
                  primaryTypographyProps={{
                    variant: 'body2',
                    sx: { wordBreak: 'break-word' },
                  }}
                  secondaryTypographyProps={{
                    variant: 'caption',
                  }}
                />
              </ListItem>
            ))}
            <div ref={listEndRef} />
          </List>
        )}
      </Box>
    </Paper>
  );
};

export default EventLog;
