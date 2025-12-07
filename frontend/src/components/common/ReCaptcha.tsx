/**
 * ReCaptcha component - Integrates Google reCAPTCHA v3 (invisible)
 * 
 * This component loads the reCAPTCHA script and provides a hook to
 * execute CAPTCHA verification. For v3, verification happens invisibly
 * in the background.
 */
import React, { useEffect, useCallback, useState } from 'react';
import { Box, Typography } from '@mui/material';
import { useQuery } from '@tanstack/react-query';
import { oauthService } from '../../services/oauthService';

// Declare global grecaptcha type
declare global {
  interface Window {
    grecaptcha: {
      ready: (callback: () => void) => void;
      execute: (siteKey: string, options: { action: string }) => Promise<string>;
      render: (container: string | HTMLElement, options: object) => number;
      reset: (widgetId?: number) => void;
    };
    hcaptcha: {
      execute: (widgetId: string, options?: object) => Promise<{ response: string }>;
      render: (container: string | HTMLElement, options: object) => string;
      reset: (widgetId?: string) => void;
    };
    onRecaptchaLoad?: () => void;
  }
}

interface ReCaptchaProps {
  /** Called when CAPTCHA token is obtained */
  onVerify: (token: string) => void;
  /** Called on CAPTCHA error */
  onError?: (error: Error) => void;
  /** Action name for reCAPTCHA v3 scoring */
  action?: string;
  /** Whether to auto-execute on mount (for invisible CAPTCHA) */
  autoExecute?: boolean;
  /** Reference to expose execute function */
  executeRef?: React.MutableRefObject<(() => Promise<string | null>) | null>;
}

/**
 * Hook to use CAPTCHA functionality
 */
export const useCaptcha = () => {
  const [isLoaded, setIsLoaded] = useState(false);
  const [isExecuting, setIsExecuting] = useState(false);

  // Fetch CAPTCHA configuration
  const { data: config, isLoading: configLoading } = useQuery({
    queryKey: ['captcha-config'],
    queryFn: oauthService.getCaptchaConfig,
    staleTime: 300000, // 5 minutes
  });

  // Load CAPTCHA script
  useEffect(() => {
    if (!config?.enabled || !config.site_key) {
      setIsLoaded(true); // Mark as loaded even if disabled
      return;
    }

    // Check if already loaded
    if (config.provider === 'recaptcha' && window.grecaptcha) {
      setIsLoaded(true);
      return;
    }
    if (config.provider === 'hcaptcha' && window.hcaptcha) {
      setIsLoaded(true);
      return;
    }

    // Create script element
    const script = document.createElement('script');
    
    if (config.provider === 'recaptcha') {
      script.src = `https://www.google.com/recaptcha/api.js?render=${config.site_key}`;
      script.onload = () => {
        window.grecaptcha.ready(() => {
          setIsLoaded(true);
        });
      };
    } else if (config.provider === 'hcaptcha') {
      script.src = 'https://js.hcaptcha.com/1/api.js?render=explicit';
      script.onload = () => {
        setIsLoaded(true);
      };
    }

    script.async = true;
    script.defer = true;
    document.head.appendChild(script);

    return () => {
      // Cleanup script if needed
    };
  }, [config]);

  // Execute CAPTCHA
  const execute = useCallback(async (action: string = 'submit'): Promise<string | null> => {
    // If CAPTCHA is not enabled, return null (no token needed)
    if (!config?.enabled || !config.site_key) {
      return null;
    }

    // Wait for script to load
    if (!isLoaded) {
      console.warn('CAPTCHA not loaded yet');
      return null;
    }

    setIsExecuting(true);

    try {
      if (config.provider === 'recaptcha') {
        const token = await window.grecaptcha.execute(config.site_key, { action });
        return token;
      } else if (config.provider === 'hcaptcha') {
        // For hCaptcha, we'd need a different approach since it's not invisible by default
        // This is a placeholder for hCaptcha support
        console.warn('hCaptcha v3-style invisible execution not supported');
        return null;
      }
      return null;
    } catch (error) {
      console.error('CAPTCHA execution failed:', error);
      throw error;
    } finally {
      setIsExecuting(false);
    }
  }, [config, isLoaded]);

  return {
    execute,
    isLoaded,
    isExecuting,
    isEnabled: config?.enabled || false,
    isLoading: configLoading,
    provider: config?.provider,
    siteKey: config?.site_key,
  };
};

/**
 * ReCaptcha component for use in forms
 * 
 * For reCAPTCHA v3, this component is invisible and automatically
 * executes when the form is submitted.
 */
const ReCaptcha: React.FC<ReCaptchaProps> = ({
  onVerify,
  onError,
  action = 'submit',
  autoExecute = false,
  executeRef,
}) => {
  const { execute, isLoaded, isEnabled, isLoading } = useCaptcha();

  // Expose execute function via ref
  useEffect(() => {
    if (executeRef) {
      executeRef.current = execute;
    }
  }, [execute, executeRef]);

  // Auto-execute if requested
  useEffect(() => {
    if (autoExecute && isLoaded && isEnabled) {
      execute(action)
        .then((token) => {
          if (token) {
            onVerify(token);
          }
        })
        .catch((error) => {
          onError?.(error);
        });
    }
  }, [autoExecute, isLoaded, isEnabled, execute, action, onVerify, onError]);

  // Show loading state if CAPTCHA is enabled but not yet loaded
  if (isLoading) {
    return null; // Don't show anything while loading config
  }

  // Don't render anything for invisible v3 CAPTCHA
  if (!isEnabled) {
    return null;
  }

  // For v3, we don't show any UI - it's completely invisible
  // The badge is handled by the script automatically
  return null;
};

/**
 * CAPTCHA badge notice (for v3 compliance)
 * 
 * reCAPTCHA v3 requires a notice about its use. This component
 * provides a compact notice that can be placed in the form.
 */
export const CaptchaNotice: React.FC = () => {
  const { isEnabled, provider } = useCaptcha();

  if (!isEnabled) {
    return null;
  }

  return (
    <Box sx={{ mt: 2, textAlign: 'center' }}>
      <Typography variant="caption" color="text.secondary">
        Protected by {provider === 'recaptcha' ? 'reCAPTCHA' : 'hCaptcha'}.{' '}
        {provider === 'recaptcha' && (
          <>
            <a
              href="https://policies.google.com/privacy"
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: 'inherit' }}
            >
              Privacy
            </a>
            {' · '}
            <a
              href="https://policies.google.com/terms"
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: 'inherit' }}
            >
              Terms
            </a>
          </>
        )}
      </Typography>
    </Box>
  );
};

export default ReCaptcha;
