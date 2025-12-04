import { Box, Container, Typography, Link } from '@mui/material';

export default function Footer() {
  return (
    <Box
      component="footer"
      sx={{
        py: 2,
        px: 2,
        mt: 'auto',
        backgroundColor: 'background.paper',
        borderTop: 1,
        borderColor: 'divider',
      }}
    >
      <Container maxWidth="lg">
        <Box
          sx={{
            display: 'flex',
            flexDirection: { xs: 'column', sm: 'row' },
            justifyContent: { xs: 'center', sm: 'space-between' },
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: { xs: 1, sm: 2 },
            textAlign: { xs: 'center', sm: 'left' },
          }}
        >
          <Typography variant="body2" color="text.secondary" sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' } }}>
            © {new Date().getFullYear()} Protein Prediction Platform | QCPP + UBF System
          </Typography>
          
          <Box sx={{ display: 'flex', gap: { xs: 1, sm: 2 }, flexWrap: 'wrap', justifyContent: 'center' }}>
            <Link
              href="https://github.com/yourusername/PP"
              target="_blank"
              rel="noopener noreferrer"
              variant="body2"
              color="text.secondary"
              underline="hover"
              sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' } }}
            >
              Documentation
            </Link>
            <Link
              href="https://github.com/yourusername/PP/issues"
              target="_blank"
              rel="noopener noreferrer"
              variant="body2"
              color="text.secondary"
              underline="hover"
              sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' } }}
            >
              Report Issue
            </Link>
            <Typography variant="body2" color="text.secondary" sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' } }}>
              v1.0.0
            </Typography>
          </Box>
        </Box>
      </Container>
    </Box>
  );
}
