import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

/**
 * Custom hook for global keyboard shortcuts
 * 
 * Supported shortcuts:
 * - Ctrl/Cmd + N: New prediction
 * - Ctrl/Cmd + K: Focus search (if available)
 * - Ctrl/Cmd + H: Go to home/dashboard
 * - Ctrl/Cmd + L: Go to history
 * - Ctrl/Cmd + S: Go to settings
 * - Escape: Close modals/dialogs
 */
export function useKeyboardShortcuts() {
  const navigate = useNavigate();

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Check for Ctrl (Windows/Linux) or Cmd (Mac)
      const isModifier = event.ctrlKey || event.metaKey;

      if (!isModifier) return;

      // Prevent default browser shortcuts
      switch (event.key.toLowerCase()) {
        case 'n':
          // Ctrl/Cmd + N: New prediction
          event.preventDefault();
          navigate('/dashboard/predictions/new');
          break;

        case 'h':
          // Ctrl/Cmd + H: Home/Dashboard
          event.preventDefault();
          navigate('/dashboard');
          break;

        case 'l':
          // Ctrl/Cmd + L: History
          event.preventDefault();
          navigate('/dashboard/history');
          break;

        case ',':
          // Ctrl/Cmd + ,: Settings (common pattern)
          event.preventDefault();
          navigate('/dashboard/settings');
          break;

        case 'k':
          // Ctrl/Cmd + K: Focus search
          event.preventDefault();
          // Dispatch custom event for search focus
          window.dispatchEvent(new CustomEvent('focusSearch'));
          break;

        default:
          break;
      }
    };

    // Add event listener
    window.addEventListener('keydown', handleKeyDown);

    // Cleanup
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [navigate]);
}

/**
 * Hook for component-specific keyboard shortcuts
 */
export function useKeyboardShortcut(
  key: string,
  callback: () => void,
  options: {
    ctrl?: boolean;
    shift?: boolean;
    alt?: boolean;
    meta?: boolean;
  } = {}
) {
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const { ctrl = false, shift = false, alt = false, meta = false } = options;

      // Check if modifiers match
      const ctrlMatch = ctrl ? event.ctrlKey : !event.ctrlKey;
      const shiftMatch = shift ? event.shiftKey : !event.shiftKey;
      const altMatch = alt ? event.altKey : !event.altKey;
      const metaMatch = meta ? event.metaKey : !event.metaKey;

      // Check if key matches (case-insensitive)
      const keyMatch = event.key.toLowerCase() === key.toLowerCase();

      if (keyMatch && ctrlMatch && shiftMatch && altMatch && metaMatch) {
        event.preventDefault();
        callback();
      }
    };

    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [key, callback, options]);
}

/**
 * Keyboard shortcut information for help dialog
 */
export const KEYBOARD_SHORTCUTS = [
  {
    category: 'Navigation',
    shortcuts: [
      { keys: ['Ctrl', 'N'], mac: ['⌘', 'N'], description: 'New prediction' },
      { keys: ['Ctrl', 'H'], mac: ['⌘', 'H'], description: 'Go to dashboard' },
      { keys: ['Ctrl', 'L'], mac: ['⌘', 'L'], description: 'View history' },
      { keys: ['Ctrl', ','], mac: ['⌘', ','], description: 'Open settings' },
    ],
  },
  {
    category: 'Search',
    shortcuts: [
      { keys: ['Ctrl', 'K'], mac: ['⌘', 'K'], description: 'Focus search' },
    ],
  },
  {
    category: 'General',
    shortcuts: [
      { keys: ['Escape'], mac: ['Escape'], description: 'Close modals' },
    ],
  },
] as const;
