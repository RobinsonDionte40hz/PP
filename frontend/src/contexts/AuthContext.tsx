/**
 * Authentication Context for managing global auth state
 */
import React, { createContext, useState, useEffect, useCallback, useRef } from 'react';
import type { ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';
import * as authService from '../services/authService';
import type { User, LoginRequest, RegisterRequest, AuthError } from '../types/auth';

interface AuthContextType {
  // State
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  
  // Actions
  login: (credentials: LoginRequest) => Promise<void>;
  register: (userData: RegisterRequest) => Promise<void>;
  logout: () => Promise<void>;
  clearError: () => void;
  refreshToken: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();
  const refreshTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Initialize auth state from localStorage on mount
  useEffect(() => {
    const initializeAuth = () => {
      try {
        const storedUser = authService.getCurrentUser();
        const isAuth = authService.isAuthenticated();
        
        if (isAuth && storedUser) {
          setUser(storedUser);
        } else {
          // Clear any stale data
          localStorage.removeItem('auth_token');
          localStorage.removeItem('refresh_token');
          localStorage.removeItem('user');
        }
      } catch (err) {
        console.error('Failed to initialize auth:', err);
        // Clear corrupted data
        localStorage.removeItem('auth_token');
        localStorage.removeItem('refresh_token');
        localStorage.removeItem('user');
      } finally {
        setIsLoading(false);
      }
    };

    initializeAuth();
  }, []);

  // Token refresh scheduling
  const scheduleTokenRefresh = useCallback((expiresIn: number) => {
    // Cancel any existing timeout
    if (refreshTimeoutRef.current) {
      clearTimeout(refreshTimeoutRef.current);
    }

    // Refresh token 5 minutes before it expires (or at 80% of lifetime)
    const refreshTime = Math.max((expiresIn * 0.8) * 1000, (expiresIn * 1000) - 300000);
    
    refreshTimeoutRef.current = setTimeout(() => {
      refreshToken();
    }, refreshTime);

    console.log(`Token refresh scheduled in ${refreshTime / 1000} seconds`);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Refresh token function
  const refreshToken = useCallback(async () => {
    try {
      const response = await authService.refreshToken();
      
      // Schedule next refresh
      scheduleTokenRefresh(response.expires_in);
      
      console.log('Token refreshed successfully');
    } catch (err) {
      console.error('Token refresh failed:', err);
      // If refresh fails, logout user
      setUser(null);
      localStorage.removeItem('auth_token');
      localStorage.removeItem('refresh_token');
      localStorage.removeItem('user');
      navigate('/login');
    }
  }, [navigate, scheduleTokenRefresh]);

  // Login function
  const login = useCallback(async (credentials: LoginRequest) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await authService.login(credentials);
      setUser(response.user);
      
      // Schedule token refresh before expiration
      scheduleTokenRefresh(response.expires_in);
      
      // Success - navigation will be handled by the caller
    } catch (err) {
      const authError = err as AuthError;
      setError(authError.detail || 'Login failed');
      throw authError;
    } finally {
      setIsLoading(false);
    }
  }, [scheduleTokenRefresh]);

  // Register function
  const register = useCallback(async (userData: RegisterRequest) => {
    setIsLoading(true);
    setError(null);
    
    try {
      await authService.register(userData);
      // Don't set user yet - let the caller handle auto-login
    } catch (err) {
      const authError = err as AuthError;
      setError(authError.detail || 'Registration failed');
      throw authError;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Logout function
  const logout = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      await authService.logout();
    } catch (err) {
      console.error('Logout error:', err);
      // Continue with local cleanup even if API fails
    } finally {
      setUser(null);
      setIsLoading(false);
      
      // Cancel any scheduled token refresh
      if (refreshTimeoutRef.current) {
        clearTimeout(refreshTimeoutRef.current);
        refreshTimeoutRef.current = null;
      }
      
      // Redirect to login
      navigate('/login');
    }
  }, [navigate]);

  // Clear error function
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  // Setup token refresh on mount if user is authenticated
  useEffect(() => {
    if (user && authService.isAuthenticated()) {
      // Default refresh in 20 minutes if expires_in not available
      scheduleTokenRefresh(1800); // 30 minutes default, refresh at 24 minutes
    }

    return () => {
      if (refreshTimeoutRef.current) {
        clearTimeout(refreshTimeoutRef.current);
      }
    };
  }, [user, scheduleTokenRefresh]);

  const value: AuthContextType = {
    user,
    isAuthenticated: !!user && authService.isAuthenticated(),
    isLoading,
    error,
    login,
    register,
    logout,
    clearError,
    refreshToken,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export default AuthContext;
