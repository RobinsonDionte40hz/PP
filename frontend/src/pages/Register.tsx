import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
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
} from '@mui/icons-material';
import { useAuth } from '../hooks/useAuth';
import { useNotification } from '../hooks/useNotification';
import { getAuthErrorMessage } from '../utils/authErrors';
import type { RegisterRequest } from '../types/auth';

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

interface RegisterFormData extends RegisterRequest {
  confirmPassword: string;
}

const Register: React.FC = () => {
  const navigate = useNavigate();
  const { register, login, isLoading } = useAuth();
  const { showSuccess, showError, showInfo, showWarning } = useNotification();
  
  // Form state
  const [formData, setFormData] = useState<RegisterFormData>({
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
  });
  
  // UI state
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  const error = localError;
  
  // Clear error/success messages after a delay
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setLocalError(null), 8000);
      return () => clearTimeout(timer);
    }
  }, [error]);
  
  useEffect(() => {
    if (success) {
      const timer = setTimeout(() => setSuccess(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [success]);
  
  // Validation state
  const [touched, setTouched] = useState({
    username: false,
    email: false,
    password: false,
    confirmPassword: false,
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

  const validateEmail = (email: string): string | null => {
    // Email is optional
    if (!email.trim()) {
      return null;
    }
    
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return 'Please enter a valid email address';
    }
    if (email.length > 255) {
      return 'Email must be less than 255 characters';
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
    if (password.length > 128) {
      return 'Password must be less than 128 characters';
    }
    
    // Check complexity
    const hasUpperCase = /[A-Z]/.test(password);
    const hasLowerCase = /[a-z]/.test(password);
    const hasNumber = /[0-9]/.test(password);
    const hasSpecialChar = /[!@#$%^&*()_+\-=[\]{};':"\\|,.<>/?]/.test(password);
    
    const complexityCount = [hasUpperCase, hasLowerCase, hasNumber, hasSpecialChar].filter(Boolean).length;
    
    if (complexityCount < 3) {
      return 'Password must contain at least 3 of: uppercase, lowercase, number, special character';
    }
    
    return null;
  };

  const validateConfirmPassword = (confirmPassword: string, password: string): string | null => {
    if (!confirmPassword) {
      return 'Please confirm your password';
    }
    if (confirmPassword !== password) {
      return 'Passwords do not match';
    }
    return null;
  };

  // Field-level errors
  const usernameError = touched.username ? validateUsername(formData.username) : null;
  const emailError = touched.email ? validateEmail(formData.email || '') : null;
  const passwordError = touched.password ? validatePassword(formData.password) : null;
  const confirmPasswordError = touched.confirmPassword 
    ? validateConfirmPassword(formData.confirmPassword, formData.password) 
    : null;

  // Form validation
  const isFormValid = 
    !validateUsername(formData.username) && 
    !validateEmail(formData.email || '') &&
    !validatePassword(formData.password) &&
    !validateConfirmPassword(formData.confirmPassword, formData.password);

  // Handle input changes
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value,
    }));
    // Clear messages when user starts typing
    if (localError) {
      setLocalError(null);
    }
    if (success) {
      setSuccess(null);
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

  const handleToggleConfirmPasswordVisibility = () => {
    setShowConfirmPassword(prev => !prev);
  };

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Mark all fields as touched
    setTouched({
      username: true,
      email: true,
      password: true,
      confirmPassword: true,
    });

    // Validate form
    if (!isFormValid) {
      showError('Please check your input and try again.', 'Invalid Input');
      return;
    }

    setLocalError(null);
    setSuccess(null);
    setIsSubmitting(true);

    try {
      // Register user
      const registerData: RegisterRequest = {
        username: formData.username,
        password: formData.password,
        email: formData.email || undefined,
      };
      
      await register(registerData);
      
      console.log('Registration successful');
      const successMsg = 'Registration successful! Logging you in...';
      setSuccess(successMsg);
      showSuccess(successMsg, 'Welcome!');
      
      // Auto-login after successful registration
      setTimeout(async () => {
        try {
          showInfo('Logging you in...', 'Please wait');
          await login({
            username: formData.username,
            password: formData.password,
          });
          
          showSuccess(`Welcome, ${formData.username}!`, 'Login Successful');
          
          // Redirect to dashboard
          setTimeout(() => {
            navigate('/');
          }, 500);
        } catch (loginErr) {
          // If auto-login fails, redirect to login page
          console.error('Auto-login failed:', loginErr);
          const errorMessage = getAuthErrorMessage(loginErr, 'login');
          setSuccess('Registration successful! Please login.');
          showWarning(errorMessage.message || 'Please login with your new credentials.', 'Registration Complete');
          
          setTimeout(() => {
            navigate('/login');
          }, 2000);
        }
      }, 1000);
      
    } catch (err) {
      console.error('Registration error:', err);
      const errorMessage = getAuthErrorMessage(err, 'register');
      setLocalError(errorMessage.message);
      showError(errorMessage.message, errorMessage.title);
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
          : 'linear-gradient(-45deg, #47597E 0%, #DBE6FD 25%, #B2AB8C 50%, #DBE6FD 75%, #47597E 100%)',
        backgroundSize: '400% 400%',
        animation: `${gradientShift} 15s ease infinite`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 0,
        margin: 0,
      }}
    >
      {/* Floating Particles */}
      <Box
        sx={{
          position: 'absolute',
          top: '10%',
          left: '10%',
          animation: `${float} 6s ease-in-out infinite`,
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
              p: 4,
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
                transform: 'translateY(-5px)',
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
                <Box
                  component="img"
                  src="/wefold-logo.svg"
                  alt="WeFold"
                  sx={{
                    height: 80,
                    width: 'auto',
                    filter: (theme) => theme.palette.mode === 'dark' 
                      ? 'brightness(1.2)' 
                      : 'none',
                  }}
                />
              </Box>
              <Typography 
                variant="h4" 
                component="h1" 
                gutterBottom
                sx={{
                  fontWeight: 700,
                  background: (theme) => theme.palette.mode === 'dark'
                    ? 'linear-gradient(135deg, #DBE6FD 0%, #B2AB8C 100%)'
                    : 'linear-gradient(135deg, #B2AB8C 0%, #47597E 100%)',
                  backgroundClip: 'text',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                }}
              >
                Create Account
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Join us to start predicting protein structures
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
                    background: 'linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.1), transparent)',
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
                    background: 'linear-gradient(90deg, transparent, rgba(244, 143, 177, 0.1), transparent)',
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

          {/* Success Alert */}
          {success && (
            <Alert severity="success" sx={{ mb: 3 }}>
              {success}
            </Alert>
          )}

          {/* Registration Form */}
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
                helperText={usernameError || 'Choose a unique username'}
                disabled={isLoading}
                autoComplete="username"
                autoFocus
                required
              />

              {/* Email Field */}
              <TextField
                fullWidth
                label="Email"
                name="email"
                type="email"
                value={formData.email}
                onChange={handleChange}
                onBlur={() => handleBlur('email')}
                error={!!emailError}
                helperText={emailError || 'Optional - for account recovery'}
                disabled={isLoading}
                autoComplete="email"
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
                helperText={passwordError || 'At least 8 characters with 3 types (uppercase, lowercase, number, special)'}
                disabled={isLoading}
                autoComplete="new-password"
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

              {/* Confirm Password Field */}
              <TextField
                fullWidth
                label="Confirm Password"
                name="confirmPassword"
                type={showConfirmPassword ? 'text' : 'password'}
                value={formData.confirmPassword}
                onChange={handleChange}
                onBlur={() => handleBlur('confirmPassword')}
                error={!!confirmPasswordError}
                helperText={confirmPasswordError || 'Re-enter your password'}
                disabled={isLoading}
                autoComplete="new-password"
                required
                InputProps={{
                  endAdornment: (
                    <InputAdornment position="end">
                      <IconButton
                        aria-label="toggle confirm password visibility"
                        onClick={handleToggleConfirmPasswordVisibility}
                        edge="end"
                        disabled={isLoading}
                      >
                        {showConfirmPassword ? <VisibilityOff /> : <Visibility />}
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
                  background: 'linear-gradient(135deg, #B2AB8C 0%, #47597E 100%)',
                  boxShadow: '0 4px 15px rgba(178, 171, 140, 0.4)',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    background: 'linear-gradient(135deg, #47597E 0%, #B2AB8C 100%)',
                    boxShadow: '0 6px 20px rgba(178, 171, 140, 0.6)',
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
                    Creating Account...
                  </>
                ) : (
                  'Create Account'
                )}
              </Button>
            </Box>
          </form>

          {/* Login Link */}
          <Box sx={{ mt: 3, textAlign: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              Already have an account?{' '}
              <Link
                to="/login"
                style={{
                  background: 'linear-gradient(135deg, #B2AB8C 0%, #47597E 100%)',
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
                Login here
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

export default Register;
