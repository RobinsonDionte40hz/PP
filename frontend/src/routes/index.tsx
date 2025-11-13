import { createBrowserRouter } from 'react-router-dom';

// Placeholder components - will be implemented later
const Dashboard = () => <div>Dashboard</div>;
const PredictionForm = () => <div>Prediction Form</div>;
const LiveMonitoring = () => <div>Live Monitoring</div>;
const ResultsAnalysis = () => <div>Results Analysis</div>;
const CampaignManagement = () => <div>Campaign Management</div>;
const HistoryBrowser = () => <div>History Browser</div>;
const Settings = () => <div>Settings</div>;

export const router = createBrowserRouter([
  {
    path: '/',
    element: <Dashboard />,
  },
  {
    path: '/predict',
    element: <PredictionForm />,
  },
  {
    path: '/monitor/:id',
    element: <LiveMonitoring />,
  },
  {
    path: '/results/:id',
    element: <ResultsAnalysis />,
  },
  {
    path: '/campaigns',
    element: <CampaignManagement />,
  },
  {
    path: '/history',
    element: <HistoryBrowser />,
  },
  {
    path: '/settings',
    element: <Settings />,
  },
]);
