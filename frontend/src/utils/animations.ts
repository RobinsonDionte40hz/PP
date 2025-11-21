import { keyframes } from '@mui/material';

/**
 * Common animation keyframes
 */

export const fadeIn = keyframes`
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
`;

export const fadeInUp = keyframes`
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
`;

export const fadeInDown = keyframes`
  from {
    opacity: 0;
    transform: translateY(-20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
`;

export const slideInLeft = keyframes`
  from {
    opacity: 0;
    transform: translateX(-20px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
`;

export const slideInRight = keyframes`
  from {
    opacity: 0;
    transform: translateX(20px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
`;

export const scaleIn = keyframes`
  from {
    opacity: 0;
    transform: scale(0.9);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
`;

export const pulse = keyframes`
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
`;

export const bounce = keyframes`
  0%, 100% {
    transform: translateY(0);
  }
  50% {
    transform: translateY(-10px);
  }
`;

/**
 * Common transition configurations
 */

export const transitions = {
  // Fast transitions for micro-interactions
  fast: {
    duration: 150,
    easing: 'cubic-bezier(0.4, 0, 0.2, 1)',
  },
  
  // Standard transitions for most UI elements
  standard: {
    duration: 300,
    easing: 'cubic-bezier(0.4, 0, 0.2, 1)',
  },
  
  // Slow transitions for complex animations
  slow: {
    duration: 500,
    easing: 'cubic-bezier(0.4, 0, 0.2, 1)',
  },
  
  // Elastic bounce effect
  elastic: {
    duration: 400,
    easing: 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
  },
  
  // Smooth spring effect
  spring: {
    duration: 350,
    easing: 'cubic-bezier(0.34, 1.56, 0.64, 1)',
  },
} as const;

/**
 * Common SX prop combinations for animations
 */

export const animationStyles = {
  fadeIn: {
    animation: `${fadeIn} 0.3s ease-in-out`,
  },
  
  fadeInUp: {
    animation: `${fadeInUp} 0.4s ease-out`,
  },
  
  fadeInDown: {
    animation: `${fadeInDown} 0.4s ease-out`,
  },
  
  slideInLeft: {
    animation: `${slideInLeft} 0.3s ease-out`,
  },
  
  slideInRight: {
    animation: `${slideInRight} 0.3s ease-out`,
  },
  
  scaleIn: {
    animation: `${scaleIn} 0.3s ease-out`,
  },
  
  // Hover effects
  hoverLift: {
    transition: 'all 0.2s ease-in-out',
    '&:hover': {
      transform: 'translateY(-4px)',
      boxShadow: 3,
    },
  },
  
  hoverScale: {
    transition: 'transform 0.2s ease-in-out',
    '&:hover': {
      transform: 'scale(1.05)',
    },
  },
  
  hoverGlow: {
    transition: 'box-shadow 0.2s ease-in-out',
    '&:hover': {
      boxShadow: '0 0 20px rgba(25, 118, 210, 0.4)',
    },
  },
  
  // Interactive states
  clickable: {
    cursor: 'pointer',
    transition: 'all 0.15s ease-in-out',
    '&:hover': {
      opacity: 0.8,
      transform: 'scale(0.98)',
    },
    '&:active': {
      transform: 'scale(0.95)',
    },
  },
  
  // Loading states
  pulse: {
    animation: `${pulse} 1.5s ease-in-out infinite`,
  },
  
  // Stagger animations (for lists)
  staggerItem: (index: number) => ({
    animation: `${fadeInUp} 0.4s ease-out`,
    animationDelay: `${index * 0.05}s`,
    animationFillMode: 'both',
  }),
} as const;

/**
 * Page transition variants for React Router
 */

export const pageTransition = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 },
  transition: { duration: 0.3 },
};
