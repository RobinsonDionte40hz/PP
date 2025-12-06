import React, { useState } from 'react';
import {
  Box,
  Container,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  TextField,
  Stack,
  Chip,
  IconButton,
  Link,
  useTheme,
  alpha,
  Paper,
} from '@mui/material';
import {
  Science as ScienceIcon,
  Speed as SpeedIcon,
  Psychology as PsychologyIcon,
  Biotech as BiotechIcon,
  ArrowForward as ArrowForwardIcon,
  PlayArrow as PlayIcon,
  Settings as SettingsIcon,
  Assessment as AssessmentIcon,
  Visibility as VisibilityIcon,
  Email as EmailIcon,
  GitHub as GitHubIcon,
  Description as DocsIcon,
  Send as SendIcon,
  CheckCircle as CheckIcon,
  LooksOne as OneIcon,
  LooksTwo as TwoIcon,
  Looks3 as ThreeIcon,
  Looks4 as FourIcon,
  AutoAwesome as AutoAwesomeIcon,
  Menu as MenuIcon,
  Close as CloseIcon,
  DarkMode as DarkModeIcon,
  LightMode as LightModeIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { keyframes } from '@mui/system';
import { useThemeStore } from '../store/themeStore';
import { useAuth } from '../hooks/useAuth';

// Animations
const fadeInUp = keyframes`
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
`;

const float = keyframes`
  0%, 100% { transform: translateY(0px) rotate(0deg); }
  50% { transform: translateY(-20px) rotate(5deg); }
`;

const gradientShift = keyframes`
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
`;

const pulse = keyframes`
  0%, 100% { opacity: 0.6; transform: scale(1); }
  50% { opacity: 1; transform: scale(1.05); }
`;

const LandingPage: React.FC = () => {
  const theme = useTheme();
  const navigate = useNavigate();
  const { mode, toggleTheme } = useThemeStore();
  const { isAuthenticated } = useAuth();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [contactForm, setContactForm] = useState({
    name: '',
    email: '',
    message: '',
  });
  const [contactSubmitted, setContactSubmitted] = useState(false);
  const [contactLoading, setContactLoading] = useState(false);
  const [contactError, setContactError] = useState<string | null>(null);

  const handleContactSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setContactLoading(true);
    setContactError(null);
    
    try {
      const response = await fetch('/api/contact', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(contactForm),
      });
      
      if (!response.ok) {
        throw new Error('Failed to send message');
      }
      
      setContactSubmitted(true);
      setTimeout(() => {
        setContactForm({ name: '', email: '', message: '' });
        setContactSubmitted(false);
      }, 3000);
    } catch {
      setContactError('Failed to send message. Please try again or email us directly.');
    } finally {
      setContactLoading(false);
    }
  };

  const workflowSteps = [
    {
      icon: <OneIcon sx={{ fontSize: 40 }} />,
      title: 'Submit Sequence',
      description: 'Enter your protein sequence or upload a FASTA file. Configure prediction parameters like iterations, agents, and accuracy level.',
      color: '#293B5F',
    },
    {
      icon: <TwoIcon sx={{ fontSize: 40 }} />,
      title: 'Multi-Agent Exploration',
      description: 'The UBF system deploys coordinated agents with adaptive strategies to efficiently explore conformational space.',
      color: '#47597E',
    },
    {
      icon: <ThreeIcon sx={{ fontSize: 40 }} />,
      title: 'Quantum Refinement',
      description: 'QCPP analyzes quantum coherence patterns, THz spectra, and golden ratio geometry. Optional two-stage refinement achieves 45-58% RMSD improvement.',
      color: '#6B7D9E',
    },
    {
      icon: <FourIcon sx={{ fontSize: 40 }} />,
      title: 'Analyze Results',
      description: 'Explore RMSD, GDT-TS, TM-Score, energy breakdown, geometric patterns, and trajectory data. Export PDB structures for downstream analysis.',
      color: '#B2AB8C',
    },
  ];

  const features = [
    {
      icon: <PsychologyIcon sx={{ fontSize: 48 }} />,
      title: 'Multi-Agent Exploration',
      description: 'Coordinated agents with adaptive behaviors navigate energy landscapes to discover optimal protein conformations.',
    },
    {
      icon: <ScienceIcon sx={{ fontSize: 48 }} />,
      title: 'Quantum Coherence Analysis',
      description: 'QCPP engine evaluates quantum coherence patterns, THz spectra, and golden ratio geometry for structure refinement.',
    },
    {
      icon: <SpeedIcon sx={{ fontSize: 48 }} />,
      title: 'Real-Time Monitoring',
      description: 'Watch predictions unfold with live energy charts, RMSD tracking, convergence metrics, and detailed event logs.',
    },
    {
      icon: <BiotechIcon sx={{ fontSize: 48 }} />,
      title: '3D Visualization & Analysis',
      description: 'Interactive NGL viewer with geometric overlays, trajectory playback, Platonic solid similarity scores, and PDB export.',
    },
  ];

  const documentationLinks = [
    { title: 'Getting Started', description: 'Quick setup guide and first prediction', icon: <PlayIcon /> },
    { title: 'API Reference', description: 'Complete REST API documentation', icon: <DocsIcon /> },
    { title: 'Configuration', description: 'Parameter tuning and optimization', icon: <SettingsIcon /> },
    { title: 'Interpreting Results', description: 'Understanding metrics and outputs', icon: <AssessmentIcon /> },
  ];

  return (
    <Box
      sx={{
        minHeight: '100vh',
        position: 'relative',
        overflow: 'hidden',
        background: (theme) => theme.palette.mode === 'dark' 
          ? 'linear-gradient(-45deg, #293B5F 0%, #47597E 25%, #293B5F 50%, #47597E 75%, #293B5F 100%)'
          : 'linear-gradient(-45deg, #47597E 0%, #1C1C1C 25%, #B2AB8C 50%, #1C1C1C 75%, #47597E 100%)',
        backgroundSize: '400% 400%',
        animation: `${gradientShift} 15s ease infinite`,
      }}
    >
      {/* Floating Particles */}
      <Box
        sx={{
          position: 'fixed',
          top: '10%',
          left: '5%',
          animation: `${float} 6s ease-in-out infinite`,
          zIndex: 0,
          pointerEvents: 'none',
        }}
      >
        <ScienceIcon sx={{ fontSize: 60, color: 'rgba(255, 255, 255, 0.15)' }} />
      </Box>
      <Box
        sx={{
          position: 'fixed',
          top: '15%',
          right: '10%',
          animation: `${float} 8s ease-in-out infinite 1s`,
          zIndex: 0,
          pointerEvents: 'none',
        }}
      >
        <BiotechIcon sx={{ fontSize: 80, color: 'rgba(255, 255, 255, 0.1)' }} />
      </Box>
      <Box
        sx={{
          position: 'fixed',
          bottom: '20%',
          left: '8%',
          animation: `${float} 7s ease-in-out infinite 2s`,
          zIndex: 0,
          pointerEvents: 'none',
        }}
      >
        <AutoAwesomeIcon sx={{ fontSize: 50, color: 'rgba(255, 255, 255, 0.12)' }} />
      </Box>
      <Box
        sx={{
          position: 'fixed',
          bottom: '30%',
          right: '5%',
          animation: `${float} 9s ease-in-out infinite 0.5s`,
          zIndex: 0,
          pointerEvents: 'none',
        }}
      >
        <ScienceIcon sx={{ fontSize: 70, color: 'rgba(255, 255, 255, 0.08)' }} />
      </Box>
      <Box
        sx={{
          position: 'fixed',
          top: '40%',
          left: '15%',
          animation: `${float} 10s ease-in-out infinite 3s`,
          zIndex: 0,
          pointerEvents: 'none',
        }}
      >
        <PsychologyIcon sx={{ fontSize: 55, color: 'rgba(255, 255, 255, 0.1)' }} />
      </Box>
      <Box
        sx={{
          position: 'fixed',
          top: '60%',
          right: '15%',
          animation: `${float} 8s ease-in-out infinite 4s`,
          zIndex: 0,
          pointerEvents: 'none',
        }}
      >
        <BiotechIcon sx={{ fontSize: 45, color: 'rgba(255, 255, 255, 0.08)' }} />
      </Box>

      {/* Navigation Bar */}
      <Box
        sx={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          zIndex: 1100,
          bgcolor: alpha(theme.palette.background.paper, 0.9),
          backdropFilter: 'blur(10px)',
          borderBottom: `1px solid ${theme.palette.divider}`,
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)',
        }}
      >
        <Container maxWidth="lg">
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', py: 1.5 }}>
            <Typography
              variant="h6"
              sx={{
                fontWeight: 700,
                background: 'linear-gradient(135deg, #293B5F 0%, #47597E 50%, #B2AB8C 100%)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                cursor: 'pointer',
                fontSize: { xs: '1.1rem', sm: '1.25rem' },
              }}
              onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
            >
              EmergentFolds
            </Typography>
            {/* Desktop Navigation */}
            <Box sx={{ display: { xs: 'none', md: 'flex' }, alignItems: 'center', gap: 2 }}>
              <Link href="#workflow" color="inherit" underline="hover" sx={{ fontWeight: 500 }}>
                How It Works
              </Link>
              <Link href="#features" color="inherit" underline="hover" sx={{ fontWeight: 500 }}>
                Features
              </Link>
              <Link href="#docs" color="inherit" underline="hover" sx={{ fontWeight: 500 }}>
                Documentation
              </Link>
              <Link href="#contact" color="inherit" underline="hover" sx={{ fontWeight: 500 }}>
                Contact
              </Link>
              <IconButton onClick={toggleTheme} color="inherit" sx={{ ml: 1 }}>
                {mode === 'dark' ? <LightModeIcon /> : <DarkModeIcon />}
              </IconButton>
              {isAuthenticated ? (
                <Button variant="contained" onClick={() => navigate('/dashboard')} sx={{ ml: 1 }}>
                  Go to Dashboard
                </Button>
              ) : (
                <>
                  <Button variant="outlined" onClick={() => navigate('/login')} sx={{ ml: 1 }}>
                    Login
                  </Button>
                  <Button variant="contained" onClick={() => navigate('/register')}>
                    Get Started
                  </Button>
                </>
              )}
            </Box>
            {/* Mobile Menu Button */}
            <IconButton
              sx={{ display: { xs: 'flex', md: 'none' } }}
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              {mobileMenuOpen ? <CloseIcon /> : <MenuIcon />}
            </IconButton>
          </Box>
          {/* Mobile Navigation Menu */}
          {mobileMenuOpen && (
            <Box
              sx={{
                display: { xs: 'flex', md: 'none' },
                flexDirection: 'column',
                gap: 2,
                pb: 2,
                borderTop: `1px solid ${theme.palette.divider}`,
                pt: 2,
              }}
            >
              <Link href="#workflow" color="inherit" underline="hover" sx={{ fontWeight: 500 }} onClick={() => setMobileMenuOpen(false)}>
                How It Works
              </Link>
              <Link href="#features" color="inherit" underline="hover" sx={{ fontWeight: 500 }} onClick={() => setMobileMenuOpen(false)}>
                Features
              </Link>
              <Link href="#docs" color="inherit" underline="hover" sx={{ fontWeight: 500 }} onClick={() => setMobileMenuOpen(false)}>
                Documentation
              </Link>
              <Link href="#contact" color="inherit" underline="hover" sx={{ fontWeight: 500 }} onClick={() => setMobileMenuOpen(false)}>
                Contact
              </Link>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography variant="body2">Theme:</Typography>
                <IconButton onClick={toggleTheme} color="inherit" size="small">
                  {mode === 'dark' ? <LightModeIcon /> : <DarkModeIcon />}
                </IconButton>
              </Box>
              {isAuthenticated ? (
                <Button variant="contained" onClick={() => { navigate('/dashboard'); setMobileMenuOpen(false); }} fullWidth sx={{ mt: 1 }}>
                  Go to Dashboard
                </Button>
              ) : (
                <Stack direction="row" spacing={2} sx={{ mt: 1 }}>
                  <Button variant="outlined" onClick={() => { navigate('/login'); setMobileMenuOpen(false); }} fullWidth>
                    Login
                  </Button>
                  <Button variant="contained" onClick={() => { navigate('/register'); setMobileMenuOpen(false); }} fullWidth>
                    Get Started
                  </Button>
                </Stack>
              )}
            </Box>
          )}
        </Container>
      </Box>

      {/* Hero Section */}
      <Box
        sx={{
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          position: 'relative',
          zIndex: 1,
          pt: { xs: 12, md: 8 },
          px: { xs: 2, sm: 0 },
        }}
      >
        <Container maxWidth="lg">
          <Grid container spacing={6} alignItems="center">
            <Grid size={{ xs: 12, md: 6 }}>
              <Box sx={{ animation: `${fadeInUp} 0.8s ease-out` }}>
                <Chip
                  label="Research Platform"
                  color="primary"
                  size="small"
                  sx={{ mb: 2, fontWeight: 600 }}
                />
                <Typography
                  variant="h1"
                  sx={{
                    fontSize: { xs: '2.5rem', md: '3.5rem', lg: '4rem' },
                    fontWeight: 800,
                    lineHeight: 1.1,
                    mb: 3,
                    color: 'white',
                  }}
                >
                  Folds Emerging Within Conformational Space
                </Typography>
                <Typography
                  variant="h5"
                  sx={{ mb: 4, lineHeight: 1.6, maxWidth: 500, color: 'rgba(255, 255, 255, 0.85)', fontSize: { xs: '1.1rem', sm: '1.25rem', md: '1.5rem' } }}
                >
                  Navigate protein energy landscapes with multi-agent exploration, quantum coherence analysis, golden ratio geometry, and real-time 3D visualization—all in one unified research platform.
                </Typography>
                <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} sx={{ mb: 4 }}>
                  <Button
                    variant="contained"
                    size="large"
                    endIcon={<ArrowForwardIcon />}
                    onClick={() => navigate('/register')}
                    sx={{
                      py: 1.5,
                      px: 4,
                      fontSize: { xs: '1rem', sm: '1.1rem' },
                      fontWeight: 600,
                    }}
                  >
                    Start Predicting
                  </Button>
                  <Button
                    variant="outlined"
                    size="large"
                    startIcon={<PlayIcon />}
                    href="#workflow"
                    sx={{ py: 1.5, px: 4 }}
                  >
                    See How It Works
                  </Button>
                </Stack>
                <Stack direction={{ xs: 'column', sm: 'row' }} spacing={{ xs: 1, sm: 3 }} sx={{ color: 'rgba(255, 255, 255, 0.85)' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <CheckIcon color="success" fontSize="small" />
                    <Typography variant="body2">QCPP + UBF Systems</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <CheckIcon color="success" fontSize="small" />
                    <Typography variant="body2">Real-time Monitoring</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <CheckIcon color="success" fontSize="small" />
                    <Typography variant="body2">3D Visualization</Typography>
                  </Box>
                </Stack>
              </Box>
            </Grid>
            <Grid size={{ xs: 12, md: 6 }}>
              <Box
                sx={{
                  textAlign: 'center',
                  position: 'relative',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <Box
                  sx={{
                    animation: `${float} 4s ease-in-out infinite, ${pulse} 3s ease-in-out infinite`,
                    position: 'relative',
                  }}
                >
                  {/* White light glow behind logo - pulses with logo */}
                  <Box
                    sx={{
                      position: 'absolute',
                      top: '50%',
                      left: '50%',
                      transform: 'translate(-50%, -50%)',
                      width: '80%',
                      height: '80%',
                      borderRadius: '50%',
                      background: 'radial-gradient(circle, rgba(255, 255, 255, 0.5) 0%, rgba(255, 255, 255, 0.2) 30%, rgba(255, 255, 255, 0) 60%)',
                      filter: 'blur(40px)',
                      zIndex: 0,
                    }}
                  />
                  <img
                    src="/emergentfoldslogo.png"
                    alt="EmergentFolds Protein Structure"
                    style={{
                      maxWidth: '100%',
                      height: 'auto',
                      maxHeight: '500px',
                      filter: 'drop-shadow(0 20px 40px rgba(41, 59, 95, 0.3))',
                      position: 'relative',
                      zIndex: 1,
                    }}
                  />
                </Box>
              </Box>
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* Workflow Section */}
      <Box id="workflow" sx={{ py: 12, bgcolor: 'background.paper', boxShadow: '0 -10px 40px rgba(0, 0, 0, 0.1), 0 10px 40px rgba(0, 0, 0, 0.1)' }}>
        <Container maxWidth="lg">
          <Box sx={{ textAlign: 'center', mb: 8 }}>
            <Typography variant="h2" fontWeight={700} gutterBottom>
              How It Works
            </Typography>
            <Typography variant="h6" color="text.secondary" sx={{ maxWidth: 600, mx: 'auto' }}>
              From sequence to structure in four simple steps
            </Typography>
          </Box>
          <Grid container spacing={4}>
            {workflowSteps.map((step, index) => (
              <Grid size={{ xs: 12, sm: 6, md: 3 }} key={index}>
                <Card
                  sx={{
                    height: '100%',
                    textAlign: 'center',
                    transition: 'transform 0.3s, box-shadow 0.3s',
                    '&:hover': {
                      transform: 'translateY(-8px)',
                      boxShadow: 6,
                    },
                  }}
                >
                  <CardContent sx={{ p: 4 }}>
                    <Box
                      sx={{
                        width: 80,
                        height: 80,
                        borderRadius: '50%',
                        bgcolor: alpha(step.color, 0.1),
                        color: step.color,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        mx: 'auto',
                        mb: 3,
                      }}
                    >
                      {step.icon}
                    </Box>
                    <Typography variant="h6" fontWeight={600} gutterBottom>
                      {step.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {step.description}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Container>
      </Box>

      {/* Features Section */}
      <Box
        id="features"
        sx={{
          py: 12,
          background: `linear-gradient(180deg, ${alpha('#293B5F', 0.02)} 0%, ${alpha('#47597E', 0.05)} 100%)`,
          boxShadow: '0 -10px 40px rgba(0, 0, 0, 0.1), 0 10px 40px rgba(0, 0, 0, 0.1)',
        }}
      >
        <Container maxWidth="lg">
          <Box sx={{ textAlign: 'center', mb: 8 }}>
            <Typography variant="h2" fontWeight={700} gutterBottom>
              Powerful Features
            </Typography>
            <Typography variant="h6" color="text.secondary" sx={{ maxWidth: 600, mx: 'auto' }}>
              Advanced tools for protein structure exploration and analysis
            </Typography>
          </Box>
          <Grid container spacing={4}>
            {features.map((feature, index) => (
              <Grid size={{ xs: 12, sm: 6 }} key={index}>
                <Paper
                  sx={{
                    p: 4,
                    height: '100%',
                    display: 'flex',
                    gap: 3,
                    transition: 'transform 0.3s, box-shadow 0.3s',
                    '&:hover': {
                      transform: 'translateY(-4px)',
                      boxShadow: 4,
                    },
                  }}
                >
                  <Box sx={{ color: 'primary.main' }}>{feature.icon}</Box>
                  <Box>
                    <Typography variant="h6" fontWeight={600} gutterBottom>
                      {feature.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {feature.description}
                    </Typography>
                  </Box>
                </Paper>
              </Grid>
            ))}
          </Grid>
        </Container>
      </Box>

      {/* Documentation Section */}
      <Box id="docs" sx={{ py: 12, bgcolor: 'background.paper', boxShadow: '0 -10px 40px rgba(0, 0, 0, 0.1), 0 10px 40px rgba(0, 0, 0, 0.1)' }}>
        <Container maxWidth="lg">
          <Box sx={{ textAlign: 'center', mb: 8 }}>
            <Typography variant="h2" fontWeight={700} gutterBottom>
              Documentation
            </Typography>
            <Typography variant="h6" color="text.secondary" sx={{ maxWidth: 600, mx: 'auto' }}>
              Everything you need to get started and make the most of EmergentFolds
            </Typography>
          </Box>
          <Grid container spacing={3}>
            {documentationLinks.map((doc, index) => (
              <Grid size={{ xs: 12, sm: 6, md: 3 }} key={index}>
                <Card
                  sx={{
                    height: '100%',
                    cursor: 'pointer',
                    transition: 'all 0.3s',
                    '&:hover': {
                      transform: 'translateY(-4px)',
                      boxShadow: 4,
                      '& .doc-icon': {
                        color: 'primary.main',
                      },
                    },
                  }}
                >
                  <CardContent sx={{ p: 3 }}>
                    <Box
                      className="doc-icon"
                      sx={{
                        color: 'text.secondary',
                        mb: 2,
                        transition: 'color 0.3s',
                      }}
                    >
                      {doc.icon}
                    </Box>
                    <Typography variant="h6" fontWeight={600} gutterBottom>
                      {doc.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {doc.description}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
          <Box sx={{ textAlign: 'center', mt: 6 }}>
            <Button
              variant="outlined"
              size="large"
              startIcon={<DocsIcon />}
              href="https://github.com/RobinsonDionte40hz/PP"
              target="_blank"
            >
              View Full Documentation
            </Button>
          </Box>
        </Container>
      </Box>

      {/* Feedback Section */}
      <Box
        sx={{
          py: 12,
          background: `linear-gradient(135deg, ${alpha('#293B5F', 0.05)} 0%, ${alpha('#B2AB8C', 0.1)} 100%)`,
          boxShadow: '0 -10px 40px rgba(0, 0, 0, 0.1), 0 10px 40px rgba(0, 0, 0, 0.1)',
        }}
      >
        <Container maxWidth="md">
          <Box sx={{ textAlign: 'center', mb: 6 }}>
            <Typography variant="h2" fontWeight={700} gutterBottom>
              We Value Your Feedback
            </Typography>
            <Typography variant="h6" color="text.secondary">
              Help us improve EmergentFolds by sharing your thoughts and suggestions
            </Typography>
          </Box>
          <Card sx={{ p: 4 }}>
            <Typography variant="body1" color="text.secondary" sx={{ mb: 3, textAlign: 'center' }}>
              Have a feature request, found a bug, or want to share your experience? 
              Log in to access the feedback form on your dashboard, or contact us directly below.
            </Typography>
            <Box sx={{ textAlign: 'center' }}>
              <Button
                variant="contained"
                size="large"
                onClick={() => navigate('/login')}
                startIcon={<VisibilityIcon />}
              >
                Login to Submit Feedback
              </Button>
            </Box>
          </Card>
        </Container>
      </Box>

      {/* Contact Section */}
      <Box id="contact" sx={{ py: 12, bgcolor: 'background.paper', boxShadow: '0 -10px 40px rgba(0, 0, 0, 0.1), 0 10px 40px rgba(0, 0, 0, 0.1)' }}>
        <Container maxWidth="lg">
          <Grid container spacing={6}>
            <Grid size={{ xs: 12, md: 5 }}>
              <Typography variant="h2" fontWeight={700} gutterBottom>
                Get In Touch
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
                Have questions about EmergentFolds? Want to collaborate on research? 
                We'd love to hear from you.
              </Typography>
              <Stack spacing={3}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <IconButton
                    sx={{
                      bgcolor: alpha(theme.palette.primary.main, 0.1),
                      color: 'primary.main',
                    }}
                  >
                    <EmailIcon />
                  </IconButton>
                  <Box>
                    <Typography variant="subtitle2" fontWeight={600}>
                      Email
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      dionterobinson.biorxiv@gmail.com
                    </Typography>
                  </Box>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <IconButton
                    sx={{
                      bgcolor: alpha(theme.palette.primary.main, 0.1),
                      color: 'primary.main',
                    }}
                    href="https://github.com/RobinsonDionte40hz/PP"
                    target="_blank"
                  >
                    <GitHubIcon />
                  </IconButton>
                  <Box>
                    <Typography variant="subtitle2" fontWeight={600}>
                      GitHub
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      RobinsonDionte40hz/PP
                    </Typography>
                  </Box>
                </Box>
              </Stack>
            </Grid>
            <Grid size={{ xs: 12, md: 7 }}>
              <Card sx={{ p: 4 }}>
                {contactSubmitted ? (
                  <Box sx={{ textAlign: 'center', py: 4 }}>
                    <CheckIcon sx={{ fontSize: 64, color: 'success.main', mb: 2 }} />
                    <Typography variant="h6">Message Sent!</Typography>
                    <Typography color="text.secondary">
                      Thank you for reaching out. We'll get back to you soon.
                    </Typography>
                  </Box>
                ) : (
                  <form onSubmit={handleContactSubmit}>
                    <Stack spacing={3}>
                      {contactError && (
                        <Typography color="error" variant="body2">
                          {contactError}
                        </Typography>
                      )}
                      <TextField
                        fullWidth
                        label="Name"
                        value={contactForm.name}
                        onChange={(e) => setContactForm({ ...contactForm, name: e.target.value })}
                        required
                        disabled={contactLoading}
                      />
                      <TextField
                        fullWidth
                        label="Email"
                        type="email"
                        value={contactForm.email}
                        onChange={(e) => setContactForm({ ...contactForm, email: e.target.value })}
                        required
                        disabled={contactLoading}
                      />
                      <TextField
                        fullWidth
                        label="Message"
                        multiline
                        rows={4}
                        value={contactForm.message}
                        onChange={(e) => setContactForm({ ...contactForm, message: e.target.value })}
                        required
                        disabled={contactLoading}
                      />
                      <Button
                        type="submit"
                        variant="contained"
                        size="large"
                        endIcon={<SendIcon />}
                        disabled={contactLoading}
                      >
                        {contactLoading ? 'Sending...' : 'Send Message'}
                      </Button>
                    </Stack>
                  </form>
                )}
              </Card>
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* Footer */}
      <Box
        sx={{
          py: 6,
          bgcolor: '#293B5F',
          color: 'white',
          boxShadow: '0 -10px 40px rgba(0, 0, 0, 0.2)',
        }}
      >
        <Container maxWidth="lg">
          <Grid container spacing={4} alignItems="center">
            <Grid size={{ xs: 12, md: 4 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: { xs: 'center', md: 'flex-start' }, gap: 2, mb: 2 }}>
                <img
                  src="/emergentfoldslogo.png"
                  alt="EmergentFolds"
                  style={{ height: '40px', width: 'auto', filter: 'brightness(2)' }}
                />
              </Box>
              <Typography variant="body2" sx={{ opacity: 0.8, textAlign: { xs: 'center', md: 'left' } }}>
                Folds Emerging Within Conformational Space
              </Typography>
            </Grid>
            <Grid size={{ xs: 12, md: 4 }}>
              <Stack direction="row" spacing={{ xs: 2, sm: 3 }} flexWrap="wrap" justifyContent={{ xs: 'center', md: 'center' }} sx={{ gap: { xs: 1, sm: 0 } }}>
                <Link href="#workflow" color="inherit" underline="hover" sx={{ opacity: 0.8 }}>
                  How It Works
                </Link>
                <Link href="#features" color="inherit" underline="hover" sx={{ opacity: 0.8 }}>
                  Features
                </Link>
                <Link href="#docs" color="inherit" underline="hover" sx={{ opacity: 0.8 }}>
                  Docs
                </Link>
                <Link href="#contact" color="inherit" underline="hover" sx={{ opacity: 0.8 }}>
                  Contact
                </Link>
              </Stack>
            </Grid>
            <Grid size={{ xs: 12, md: 4 }}>
              <Typography variant="body2" sx={{ opacity: 0.6, textAlign: { xs: 'center', md: 'right' } }}>
                © {new Date().getFullYear()} EmergentFolds. All rights reserved.
              </Typography>
            </Grid>
          </Grid>
        </Container>
      </Box>
    </Box>
  );
};

export default LandingPage;
