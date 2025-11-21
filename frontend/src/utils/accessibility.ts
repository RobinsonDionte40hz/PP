/**
 * Accessibility utilities and ARIA label helpers
 */

/**
 * Generate accessible button props
 */
export function getAccessibleButtonProps(label: string, options?: {
  expanded?: boolean;
  controls?: string;
  pressed?: boolean;
}) {
  return {
    'aria-label': label,
    'aria-expanded': options?.expanded,
    'aria-controls': options?.controls,
    'aria-pressed': options?.pressed,
  };
}

/**
 * Generate accessible link props
 */
export function getAccessibleLinkProps(label: string, isExternal = false) {
  return {
    'aria-label': label,
    ...(isExternal && {
      target: '_blank',
      rel: 'noopener noreferrer',
      'aria-label': `${label} (opens in new tab)`,
    }),
  };
}

/**
 * Generate accessible form field props
 */
export function getAccessibleFormFieldProps(
  id: string,
  label: string,
  options?: {
    required?: boolean;
    invalid?: boolean;
    describedBy?: string;
  }
) {
  return {
    id,
    'aria-label': label,
    'aria-required': options?.required,
    'aria-invalid': options?.invalid,
    'aria-describedby': options?.describedBy,
  };
}

/**
 * Generate accessible dialog props
 */
export function getAccessibleDialogProps(
  title: string,
  description?: string
) {
  return {
    'aria-labelledby': `${title}-dialog-title`,
    'aria-describedby': description ? `${title}-dialog-description` : undefined,
    role: 'dialog',
  };
}

/**
 * Generate accessible alert props
 */
export function getAccessibleAlertProps(severity: 'error' | 'warning' | 'info' | 'success') {
  const roleMap = {
    error: 'alert',
    warning: 'alert',
    info: 'status',
    success: 'status',
  };

  return {
    role: roleMap[severity],
    'aria-live': severity === 'error' || severity === 'warning' ? 'assertive' : 'polite',
    'aria-atomic': true,
  };
}

/**
 * Generate accessible tab props
 */
export function getAccessibleTabProps(index: number, id: string) {
  return {
    id: `tab-${id}-${index}`,
    'aria-controls': `tabpanel-${id}-${index}`,
    role: 'tab',
  };
}

/**
 * Generate accessible tab panel props
 */
export function getAccessibleTabPanelProps(index: number, id: string, value: number) {
  return {
    id: `tabpanel-${id}-${index}`,
    'aria-labelledby': `tab-${id}-${index}`,
    role: 'tabpanel',
    hidden: value !== index,
  };
}

/**
 * Generate accessible table props
 */
export function getAccessibleTableProps(caption: string) {
  return {
    'aria-label': caption,
    role: 'table',
  };
}

/**
 * Generate accessible progress bar props
 */
export function getAccessibleProgressProps(label: string, value: number, max = 100) {
  return {
    'aria-label': label,
    'aria-valuenow': value,
    'aria-valuemin': 0,
    'aria-valuemax': max,
    'aria-valuetext': `${Math.round((value / max) * 100)}% complete`,
    role: 'progressbar',
  };
}

/**
 * Generate accessible menu props
 */
export function getAccessibleMenuProps(label: string) {
  return {
    'aria-label': label,
    role: 'menu',
  };
}

/**
 * Generate accessible menu item props
 */
export function getAccessibleMenuItemProps(label: string) {
  return {
    'aria-label': label,
    role: 'menuitem',
  };
}

/**
 * Keyboard navigation helpers
 */
export const KeyboardKeys = {
  ENTER: 'Enter',
  SPACE: ' ',
  ESCAPE: 'Escape',
  TAB: 'Tab',
  ARROW_UP: 'ArrowUp',
  ARROW_DOWN: 'ArrowDown',
  ARROW_LEFT: 'ArrowLeft',
  ARROW_RIGHT: 'ArrowRight',
  HOME: 'Home',
  END: 'End',
  PAGE_UP: 'PageUp',
  PAGE_DOWN: 'PageDown',
} as const;

/**
 * Handle keyboard navigation for list items
 */
export function handleListKeyboardNav(
  event: React.KeyboardEvent,
  currentIndex: number,
  itemCount: number,
  onSelect: (index: number) => void
) {
  switch (event.key) {
    case KeyboardKeys.ARROW_UP:
      event.preventDefault();
      onSelect(currentIndex > 0 ? currentIndex - 1 : itemCount - 1);
      break;
    case KeyboardKeys.ARROW_DOWN:
      event.preventDefault();
      onSelect(currentIndex < itemCount - 1 ? currentIndex + 1 : 0);
      break;
    case KeyboardKeys.HOME:
      event.preventDefault();
      onSelect(0);
      break;
    case KeyboardKeys.END:
      event.preventDefault();
      onSelect(itemCount - 1);
      break;
    case KeyboardKeys.ENTER:
    case KeyboardKeys.SPACE:
      event.preventDefault();
      // Trigger selection or action
      break;
  }
}

/**
 * Focus management utilities
 */
export function trapFocus(element: HTMLElement) {
  const focusableElements = element.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  );
  
  const firstElement = focusableElements[0] as HTMLElement;
  const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

  function handleTabKey(e: KeyboardEvent) {
    if (e.key !== KeyboardKeys.TAB) return;

    if (e.shiftKey) {
      if (document.activeElement === firstElement) {
        lastElement.focus();
        e.preventDefault();
      }
    } else {
      if (document.activeElement === lastElement) {
        firstElement.focus();
        e.preventDefault();
      }
    }
  }

  element.addEventListener('keydown', handleTabKey);

  return () => {
    element.removeEventListener('keydown', handleTabKey);
  };
}

/**
 * Screen reader only text (visually hidden but accessible)
 */
export const srOnly = {
  position: 'absolute',
  width: '1px',
  height: '1px',
  padding: 0,
  margin: '-1px',
  overflow: 'hidden',
  clip: 'rect(0, 0, 0, 0)',
  whiteSpace: 'nowrap',
  borderWidth: 0,
} as const;
