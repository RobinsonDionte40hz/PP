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
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: 2,
          }}
        >
          <Typography variant="body2" color="text.secondary">
            © {new Date().getFullYear()} Protein Prediction Platform | QCPP + UBF System
          </Typography>
          
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Link
              href="https://github.com/yourusername/PP"
              target="_blank"
              rel="noopener noreferrer"
              variant="body2"
              color="text.secondary"
              underline="hover"
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
            >
              Report Issue
            </Link>
            <Typography variant="body2" color="text.secondary">
              v1.0.0
            </Typography>
          </Box>
        </Box>
      </Container>
    </Box>
  );
}
