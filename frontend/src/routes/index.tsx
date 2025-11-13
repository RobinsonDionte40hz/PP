import { createBrowserRouter } from 'react-router-dom';
import AppLayout from '../components/layout/AppLayout';
import ErrorBoundary from '../components/common/ErrorBoundary';
import Dashboard from '../pages/Dashboard';
import PredictionForm from '../pages/PredictionForm';
import LiveMonitoring from '../pages/LiveMonitoring';
import ResultsAnalysis from '../pages/ResultsAnalysis';
import StructureVisualization from '../pages/StructureVisualization';
import CampaignManagement from '../pages/CampaignManagement';
import HistoryBrowser from '../pages/HistoryBrowser';
import Settings from '../pages/Settings';

export const router = createBrowserRouter([
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
        element: <Dashboard />,
      },
      {
        path: 'predict',
        element: <PredictionForm />,
      },
      {
        path: 'predictions/new',
        element: <PredictionForm />,
      },
      {
        path: 'monitor/:id',
        element: <LiveMonitoring />,
      },
      {
        path: 'monitor/active',
        element: <LiveMonitoring />,
      },
      {
        path: 'results/:id',
        element: <ResultsAnalysis />,
      },
      {
        path: 'results/latest',
        element: <ResultsAnalysis />,
      },
      {
        path: 'campaigns',
        element: <CampaignManagement />,
      },
      {
        path: 'history',
        element: <HistoryBrowser />,
      },
      {
        path: 'visualization',
        element: <StructureVisualization />,
      },
      {
        path: 'settings',
        element: <Settings />,
      },
    ],
  },
]);
