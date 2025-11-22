/**
 * Router wrapper to provide AuthContext inside router context
 */
import { Outlet } from 'react-router-dom';
import { AuthProvider } from '../contexts/AuthContext';

export const AuthWrapper = () => {
  return (
    <AuthProvider>
      <Outlet />
    </AuthProvider>
  );
};
