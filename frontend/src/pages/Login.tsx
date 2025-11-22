import React, { useState } from 'react';
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
} from '@mui/material';
import { Visibility, VisibilityOff, Login as LoginIcon } from '@mui/icons-material';
import { useAuth } from '../hooks/useAuth';
import type { LoginRequest } from '../types/auth';

const Login: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { login, isLoading, error: authError } = useAuth();
  
  // Get the location user was trying to access before being redirected
  const from = (location.state as { from?: { pathname: string } })?.from?.pathname || '/';
  
  // Form state
  const [formData, setFormData] = useState<LoginRequest>({
    username: '',
    password: '',
  });
  
  // UI state
  const [showPassword, setShowPassword] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  
  // Use auth error or local error
  const error = authError || localError;
  
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
      return;
    }

    setLocalError(null);

    try {
      await login(formData);
      
      // Success - redirect to the page user was trying to access or dashboard
      console.log('Login successful, redirecting to:', from);
      navigate(from, { replace: true });
    } catch (err) {
      console.error('Login error:', err);
      // Error is handled by AuthContext
    }
  };

  return (
    <Container maxWidth="sm">
      <Box
        sx={{
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          py: 4,
        }}
      >
        <Paper
          elevation={3}
          sx={{
            p: 4,
            width: '100%',
            borderRadius: 2,
          }}
        >
          {/* Header */}
          <Box sx={{ textAlign: 'center', mb: 4 }}>
            <LoginIcon sx={{ fontSize: 48, color: 'primary.main', mb: 1 }} />
            <Typography variant="h4" component="h1" gutterBottom>
              Welcome Back
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Sign in to access your protein predictions
            </Typography>
          </Box>

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
                disabled={isLoading || !isFormValid}
                sx={{ mt: 1 }}
              >
                {isLoading ? (
                  <>
                    <CircularProgress size={24} sx={{ mr: 1 }} />
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
                  color: 'inherit',
                  textDecoration: 'none',
                  fontWeight: 500,
                }}
              >
                Register here
              </Link>
            </Typography>
          </Box>
        </Paper>
      </Box>
    </Container>
  );
};

export default Login;
