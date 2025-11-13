import { describe, it, expect } from 'vitest';
import { render, screen } from '../test/test-utils';
import { PredictionStatusBadge, QualityBadge } from '../components/common';

describe('PredictionStatusBadge', () => {
  it('renders pending status', () => {
    render(<PredictionStatusBadge status="pending" />);
    expect(screen.getByText(/pending/i)).toBeInTheDocument();
  });

  it('renders running status', () => {
    render(<PredictionStatusBadge status="running" />);
    expect(screen.getByText(/running/i)).toBeInTheDocument();
  });

  it('renders completed status', () => {
    render(<PredictionStatusBadge status="completed" />);
    expect(screen.getByText(/completed/i)).toBeInTheDocument();
  });

  it('renders failed status', () => {
    render(<PredictionStatusBadge status="failed" />);
    expect(screen.getByText(/failed/i)).toBeInTheDocument();
  });
});

describe('QualityBadge', () => {
  it('renders excellent quality', () => {
    render(<QualityBadge quality="excellent" />);
    expect(screen.getByText(/excellent/i)).toBeInTheDocument();
  });

  it('renders good quality', () => {
    render(<QualityBadge quality="good" />);
    expect(screen.getByText(/good/i)).toBeInTheDocument();
  });

  it('renders acceptable quality', () => {
    render(<QualityBadge quality="acceptable" />);
    expect(screen.getByText(/acceptable/i)).toBeInTheDocument();
  });

  it('renders poor quality', () => {
    render(<QualityBadge quality="poor" />);
    expect(screen.getByText(/poor/i)).toBeInTheDocument();
  });
});
