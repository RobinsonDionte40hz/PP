import { Paper, Box, Typography, Chip, TextField, MenuItem } from '@mui/material';
import { useState } from 'react';

export interface LogEvent {
  id: string;
  timestamp: string;
  level: 'info' | 'warning' | 'error' | 'success';
  message: string;
  details?: string;
}

interface EventLogProps {
  events: LogEvent[];
  maxHeight?: number;
  showFilters?: boolean;
}

export default function EventLog({ events, maxHeight = 400, showFilters = true }: EventLogProps) {
  const [filterLevel, setFilterLevel] = useState<string>('all');

  const filteredEvents = filterLevel === 'all' 
    ? events 
    : events.filter(event => event.level === filterLevel);

  const getLevelColor = (level: LogEvent['level']) => {
    switch (level) {
      case 'info': return 'info';
      case 'warning': return 'warning';
      case 'error': return 'error';
      case 'success': return 'success';
      default: return 'default';
    }
  };

  return (
    <Paper variant="outlined">
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h6">Event Log</Typography>
          
          {showFilters && (
            <TextField
              select
              size="small"
              value={filterLevel}
              onChange={(e) => setFilterLevel(e.target.value)}
              sx={{ minWidth: 120 }}
            >
              <MenuItem value="all">All Events</MenuItem>
              <MenuItem value="info">Info</MenuItem>
              <MenuItem value="success">Success</MenuItem>
              <MenuItem value="warning">Warning</MenuItem>
              <MenuItem value="error">Error</MenuItem>
            </TextField>
          )}
        </Box>
      </Box>
      
      <Box
        sx={{
          maxHeight,
          overflowY: 'auto',
          p: 2,
        }}
      >
        {filteredEvents.length === 0 ? (
          <Typography variant="body2" color="text.secondary" textAlign="center" py={4}>
            No events to display
          </Typography>
        ) : (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            {filteredEvents.map((event) => (
              <Box
                key={event.id}
                sx={{
                  p: 1.5,
                  borderRadius: 1,
                  backgroundColor: 'background.default',
                  borderLeft: 4,
                  borderColor: `${getLevelColor(event.level)}.main`,
                }}
              >
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 0.5 }}>
                  <Chip
                    label={event.level.toUpperCase()}
                    size="small"
                    color={getLevelColor(event.level)}
                    sx={{ height: 20 }}
                  />
                  <Typography variant="caption" color="text.secondary">
                    {new Date(event.timestamp).toLocaleTimeString()}
                  </Typography>
                </Box>
                
                <Typography variant="body2" sx={{ mt: 1 }}>
                  {event.message}
                </Typography>
                
                {event.details && (
                  <Typography
                    variant="caption"
                    color="text.secondary"
                    sx={{
                      mt: 0.5,
                      display: 'block',
                      fontFamily: 'monospace',
                    }}
                  >
                    {event.details}
                  </Typography>
                )}
              </Box>
            ))}
          </Box>
        )}
      </Box>
    </Paper>
  );
}
