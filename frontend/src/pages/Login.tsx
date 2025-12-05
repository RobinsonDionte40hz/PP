import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation, Link } from 'react-router-dom';
import {
  Box,
  Button,
  Container,
  TextField,
  Typography,
  Paper,
  Alert,
  CircularProgress,
  InputAdornment,
  IconButton,
  Chip,
  Divider,
  keyframes,
} from '@mui/material';
import { 
  Visibility, 
  VisibilityOff, 
  Science,
  Biotech,
  AutoAwesome,
  DarkMode as DarkModeIcon,
  LightMode as LightModeIcon,
} from '@mui/icons-material';
import { useAuth } from '../hooks/useAuth';
import { useNotification } from '../hooks/useNotification';
import { useThemeStore } from '../store/themeStore';
import { getAuthErrorMessage } from '../utils/authErrors';
import type { LoginRequest } from '../types/auth';

// Animations
const float = keyframes`
  0%, 100% { transform: translateY(0px) rotate(0deg); }
  50% { transform: translateY(-20px) rotate(5deg); }
`;

const pulse = keyframes`
  0%, 100% { opacity: 0.6; transform: scale(1); }
  50% { opacity: 1; transform: scale(1.05); }
`;

const gradientShift = keyframes`
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
`;

const shimmer = keyframes`
  0% { background-position: -1000px 0; }
  100% { background-position: 1000px 0; }
`;

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

const Login: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { login, isLoading, error: authError } = useAuth();
  const { showSuccess, showError } = useNotification();
  const { mode, toggleTheme } = useThemeStore();
  
  // Get the location user was trying to access before being redirected
  const from = (location.state as { from?: { pathname: string } })?.from?.pathname || '/dashboard';
  
  // Form state
  const [formData, setFormData] = useState<LoginRequest>({
    username: '',
    password: '',
  });
  
  // UI state
  const [showPassword, setShowPassword] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  // Use auth error or local error
  const error = authError || localError;
  
  // Show toast notification when auth error changes
  useEffect(() => {
    if (authError) {
      const errorMessage = getAuthErrorMessage({ response: { data: { detail: authError } } }, 'login');
      showError(errorMessage.message, errorMessage.title);
    }
  }, [authError, showError]);
  
  // Validation state
  const [touched, setTouched] = useState({
    username: false,
    password: false,
  });

  // Validation rules
  const validateUsername = (username: string): string | null => {
    if (!username.trim()) {
      return 'Username is required';
    }
    if (username.length < 3) {
      return 'Username must be at least 3 characters';
    }
    if (username.length > 50) {
      return 'Username must be less than 50 characters';
    }
    if (!/^[a-zA-Z0-9_-]+$/.test(username)) {
      return 'Username can only contain letters, numbers, underscores, and hyphens';
    }
    return null;
  };

  const validatePassword = (password: string): string | null => {
    if (!password) {
      return 'Password is required';
    }
    if (password.length < 8) {
      return 'Password must be at least 8 characters';
    }
    return null;
  };

  // Field-level errors
  const usernameError = touched.username ? validateUsername(formData.username) : null;
  const passwordError = touched.password ? validatePassword(formData.password) : null;

  // Form validation
  const isFormValid = 
    !validateUsername(formData.username) && 
    !validatePassword(formData.password);

  // Handle input changes
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value,
    }));
    // Clear error when user starts typing
    if (localError) {
      setLocalError(null);
    }
  };

  // Handle field blur
  const handleBlur = (field: keyof typeof touched) => {
    setTouched(prev => ({
      ...prev,
      [field]: true,
    }));
  };

  // Toggle password visibility
  const handleTogglePasswordVisibility = () => {
    setShowPassword(prev => !prev);
  };

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Mark all fields as touched
    setTouched({
      username: true,
      password: true,
    });

    // Validate form
    if (!isFormValid) {
      showError('Please check your input and try again.', 'Invalid Input');
      return;
    }

    setLocalError(null);
    setIsSubmitting(true);

    try {
      await login(formData);
      
      // Success - show notification and redirect
      showSuccess(`Welcome back, ${formData.username}!`, 'Login Successful');
      console.log('Login successful, redirecting to:', from);
      
      // Small delay to show success message
      setTimeout(() => {
        navigate(from, { replace: true });
      }, 500);
    } catch (err) {
      console.error('Login error:', err);
      const errorMessage = getAuthErrorMessage(err, 'login');
      setLocalError(errorMessage.message);
      // Toast notification is shown by useEffect
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        width: '100vw',
        position: 'relative',
        overflow: 'hidden',
        background: (theme) => theme.palette.mode === 'dark' 
          ? 'linear-gradient(-45deg, #293B5F 0%, #47597E 25%, #293B5F 50%, #47597E 75%, #293B5F 100%)'
          : 'linear-gradient(-45deg, #47597E 0%, #1C1C1C 25%, #B2AB8C 50%, #1C1C1C 75%, #47597E 100%)',
        backgroundSize: '400% 400%',
        animation: `${gradientShift} 15s ease infinite`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 0,
        margin: 0,
      }}
    >
      {/* Theme Toggle Button */}
      <IconButton
        onClick={toggleTheme}
        sx={{
          position: 'fixed',
          top: 16,
          right: 16,
          zIndex: 1200,
          bgcolor: 'rgba(255, 255, 255, 0.1)',
          backdropFilter: 'blur(10px)',
          color: 'white',
          '&:hover': {
            bgcolor: 'rgba(255, 255, 255, 0.2)',
          },
        }}
      >
        {mode === 'dark' ? <LightModeIcon /> : <DarkModeIcon />}
      </IconButton>
      {/* Floating Particles - Hidden on mobile for better UX */}}
      <Box
        sx={{
          position: 'absolute',
          top: '10%',
          left: '10%',
          animation: `${float} 6s ease-in-out infinite`,
          display: { xs: 'none', md: 'block' },
        }}
      >
        <Science sx={{ fontSize: 60, color: 'rgba(255, 255, 255, 0.15)' }} />
      </Box>
      <Box
        sx={{
          position: 'absolute',
          top: '20%',
          right: '15%',
          animation: `${float} 8s ease-in-out infinite 1s`,
          display: { xs: 'none', md: 'block' },
        }}
      >
        <Biotech sx={{ fontSize: 80, color: 'rgba(255, 255, 255, 0.1)' }} />
      </Box>
      <Box
        sx={{
          position: 'absolute',
          bottom: '15%',
          left: '15%',
          animation: `${float} 7s ease-in-out infinite 2s`,
          display: { xs: 'none', md: 'block' },
        }}
      >
        <AutoAwesome sx={{ fontSize: 50, color: 'rgba(255, 255, 255, 0.12)' }} />
      </Box>
      <Box
        sx={{
          position: 'absolute',
          bottom: '25%',
          right: '10%',
          animation: `${float} 9s ease-in-out infinite 0.5s`,
          display: { xs: 'none', md: 'block' },
        }}
      >
        <Science sx={{ fontSize: 70, color: 'rgba(255, 255, 255, 0.08)' }} />
      </Box>

      <Container maxWidth="sm" sx={{ position: 'relative', zIndex: 1, width: '100%' }}>
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            py: 4,
            px: 2,
            animation: `${fadeInUp} 0.6s ease-out`,
          }}
        >
          <Paper
            elevation={24}
            sx={{
              p: { xs: 3, sm: 4 },
              width: '100%',
              borderRadius: 4,
              background: (theme) => theme.palette.mode === 'dark'
                ? 'rgba(30, 30, 30, 0.9)'
                : 'rgba(255, 255, 255, 0.95)',
              backdropFilter: 'blur(20px)',
              border: (theme) => theme.palette.mode === 'dark'
                ? '1px solid rgba(255, 255, 255, 0.1)'
                : '1px solid rgba(255, 255, 255, 0.3)',
              boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
              transition: 'transform 0.3s ease',
              '&:hover': {
                transform: { xs: 'none', sm: 'translateY(-5px)' },
              },
            }}
          >
            {/* Header */}
            <Box sx={{ textAlign: 'center', mb: 4 }}>
              {/* Logo */}
              <Box
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  mb: 3,
                  animation: `${pulse} 3s ease-in-out infinite`,
                }}
              >
                <img
                  src="/emergentfoldslogo.png"
                  alt="EmergentFolds"
                  style={{
                    height: 'auto',
                    maxHeight: '150px',
                    width: '100%',
                    maxWidth: '180px',
                    objectFit: 'contain',
                  }}
                />
              </Box>
              <Typography 
                variant="h4" 
                component="h1" 
                gutterBottom
                sx={{
                  fontWeight: 700,
                  fontSize: { xs: '1.75rem', sm: '2.125rem' },
                  background: (theme) => theme.palette.mode === 'dark'
                    ? 'linear-gradient(135deg, #1C1C1C 0%, #B2AB8C 100%)'
                    : 'linear-gradient(135deg, #293B5F 0%, #47597E 100%)',
                  backgroundClip: 'text',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                }}
              >
                Welcome Back
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Sign in to access your protein predictions
              </Typography>
              
              {/* Feature Chips */}
              <Box sx={{ display: 'flex', gap: 1, justifyContent: 'center', flexWrap: 'wrap', mt: 2 }}>
                <Chip 
                  icon={<Science />} 
                  label="QCPP Powered" 
                  size="small" 
                  variant="outlined"
                  sx={{ 
                    borderColor: 'primary.main',
                    animation: `${shimmer} 3s infinite`,
                    background: 'linear-gradient(90deg, transparent, rgba(71, 89, 126, 0.15), transparent)',
                    backgroundSize: '1000px 100%',
                  }}
                />
                <Chip 
                  icon={<Biotech />} 
                  label="UBF System" 
                  size="small" 
                  variant="outlined"
                  sx={{ 
                    borderColor: 'secondary.main',
                    animation: `${shimmer} 3s infinite 0.5s`,
                    background: 'linear-gradient(90deg, transparent, rgba(178, 171, 140, 0.15), transparent)',
                    backgroundSize: '1000px 100%',
                  }}
                />
              </Box>
            </Box>

            <Divider sx={{ mb: 3, opacity: 0.3 }} />

          {/* Error Alert */}
          {error && (
            <Alert severity="error" sx={{ mb: 3 }}>
              {error}
            </Alert>
          )}

          {/* Login Form */}
          <form onSubmit={handleSubmit} noValidate>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {/* Username Field */}
              <TextField
                fullWidth
                label="Username"
                name="username"
                value={formData.username}
                onChange={handleChange}
                onBlur={() => handleBlur('username')}
                error={!!usernameError}
                helperText={usernameError || ' '}
                disabled={isLoading}
                autoComplete="username"
                autoFocus
                required
              />

              {/* Password Field */}
              <TextField
                fullWidth
                label="Password"
                name="password"
                type={showPassword ? 'text' : 'password'}
                value={formData.password}
                onChange={handleChange}
                onBlur={() => handleBlur('password')}
                error={!!passwordError}
                helperText={passwordError || ' '}
                disabled={isLoading}
                autoComplete="current-password"
                required
                InputProps={{
                  endAdornment: (
                    <InputAdornment position="end">
                      <IconButton
                        aria-label="toggle password visibility"
                        onClick={handleTogglePasswordVisibility}
                        edge="end"
                        disabled={isLoading}
                      >
                        {showPassword ? <VisibilityOff /> : <Visibility />}
                      </IconButton>
                    </InputAdornment>
                  ),
                }}
              />

              {/* Submit Button */}
              <Button
                type="submit"
                variant="contained"
                size="large"
                fullWidth
                disabled={isLoading || isSubmitting || !isFormValid}
                sx={{ 
                  mt: 1,
                  py: 1.5,
                  fontWeight: 600,
                  fontSize: '1.1rem',
                  background: 'linear-gradient(135deg, #293B5F 0%, #47597E 100%)',
                  boxShadow: '0 4px 15px rgba(41, 59, 95, 0.4)',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    background: 'linear-gradient(135deg, #47597E 0%, #293B5F 100%)',
                    boxShadow: '0 6px 20px rgba(71, 89, 126, 0.6)',
                    transform: 'translateY(-2px)',
                  },
                  '&:active': {
                    transform: 'translateY(0)',
                  },
                  '&:disabled': {
                    background: 'linear-gradient(135deg, #ccc 0%, #999 100%)',
                    boxShadow: 'none',
                  },
                }}
              >
                {isLoading || isSubmitting ? (
                  <>
                    <CircularProgress size={24} sx={{ mr: 1 }} color="inherit" />
                    Signing In...
                  </>
                ) : (
                  'Sign In'
                )}
              </Button>
            </Box>
          </form>

          {/* Registration Link */}
          <Box sx={{ mt: 3, textAlign: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              Don't have an account?{' '}
              <Link
                to="/register"
                style={{
                  background: 'linear-gradient(135deg, #293B5F 0%, #47597E 100%)',
                  backgroundClip: 'text',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  textDecoration: 'none',
                  fontWeight: 600,
                  transition: 'all 0.3s ease',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.textDecoration = 'underline';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.textDecoration = 'none';
                }}
              >
                Register here
              </Link>
            </Typography>
          </Box>

          {/* Footer Badge */}
          <Box sx={{ mt: 4, textAlign: 'center' }}>
            <Typography 
              variant="caption" 
              color="text.secondary"
              sx={{ 
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 0.5,
                opacity: 0.7,
              }}
            >
              <AutoAwesome sx={{ fontSize: 14 }} />
              Quantum-Enhanced Protein Structure Prediction
              <AutoAwesome sx={{ fontSize: 14 }} />
            </Typography>
          </Box>
        </Paper>
      </Box>
    </Container>
    </Box>
  );
};

export default Login;
