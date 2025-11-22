import { createBrowserRouter } from 'react-router-dom';
import { lazy, Suspense } from 'react';
import AppLayout from '../components/layout/AppLayout';
import ErrorBoundary from '../components/common/ErrorBoundary';
import LoadingSpinner from '../components/common/LoadingSpinner';

// Lazy load all page components for code splitting
const Dashboard = lazy(() => import('../pages/Dashboard'));
const PredictionForm = lazy(() => import('../pages/PredictionForm'));
const LiveMonitoring = lazy(() => import('../pages/LiveMonitoring'));
const ResultsAnalysis = lazy(() => import('../pages/ResultsAnalysis'));
const StructureVisualization = lazy(() => import('../pages/StructureVisualization'));
const CampaignManagement = lazy(() => import('../pages/CampaignManagement'));
const HistoryBrowser = lazy(() => import('../pages/HistoryBrowser'));
const Settings = lazy(() => import('../pages/Settings'));
const Login = lazy(() => import('../pages/Login'));
const Register = lazy(() => import('../pages/Register'));

// Suspense wrapper for lazy-loaded components
const withSuspense = (Component: React.LazyExoticComponent<React.ComponentType<any>>) => (
  <Suspense fallback={<LoadingSpinner />}>
    <Component />
  </Suspense>
);

export const router = createBrowserRouter([
  {
    path: '/login',
    element: (
      <ErrorBoundary>
        {withSuspense(Login)}
      </ErrorBoundary>
    ),
  },
  {
    path: '/register',
    element: (
      <ErrorBoundary>
        {withSuspense(Register)}
      </ErrorBoundary>
    ),
  },
  {
    path: '/',
    element: (
      <ErrorBoundary>
        <AppLayout />
      </ErrorBoundary>
    ),
    children: [
      {
        index: true,
        element: withSuspense(Dashboard),
      },
      {
        path: 'predict',
        element: withSuspense(PredictionForm),
      },
      {
        path: 'predictions/new',
        element: withSuspense(PredictionForm),
      },
      {
        path: 'monitor/:id',
        element: withSuspense(LiveMonitoring),
      },
      {
        path: 'monitor/active',
        element: withSuspense(LiveMonitoring),
      },
      {
        path: 'results/:id',
        element: withSuspense(ResultsAnalysis),
      },
      {
        path: 'results/latest',
        element: withSuspense(ResultsAnalysis),
      },
      {
        path: 'campaigns',
        element: withSuspense(CampaignManagement),
      },
      {
        path: 'history',
        element: withSuspense(HistoryBrowser),
      },
      {
        path: 'visualization',
        element: withSuspense(StructureVisualization),
      },
      {
        path: 'settings',
        element: withSuspense(Settings),
      },
    ],
  },
]);
