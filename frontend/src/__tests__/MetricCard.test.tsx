import { describe, it, expect } from 'vitest';
import { render, screen } from '../test/test-utils';
import MetricCard from '../components/common/MetricCard';

describe('MetricCard', () => {
  it('renders metric title and value', () => {
    render(
      <MetricCard 
        title="Energy" 
        value="-123.45" 
        unit="kcal/mol" 
      />
    );
    
    expect(screen.getByText('Energy')).toBeInTheDocument();
    expect(screen.getByText('-123.45')).toBeInTheDocument();
    expect(screen.getByText('kcal/mol')).toBeInTheDocument();
  });

  it('renders with icon when provided', () => {
    const TestIcon = () => <svg data-testid="test-icon" />;
    render(
      <MetricCard 
        title="RMSD" 
        value="2.5" 
        icon={<TestIcon />} 
      />
    );
    
    expect(screen.getByTestId('test-icon')).toBeInTheDocument();
  });

  it('renders trend indicator', () => {
    render(
      <MetricCard 
        title="Energy" 
        value="-150" 
        trend="down"
        trendValue="5%"
      />
    );
    
    expect(screen.getByText('5%')).toBeInTheDocument();
  });

  it('applies color variant correctly', () => {
    const { container } = render(
      <MetricCard 
        title="Status" 
        value="Active" 
        color="success"
      />
    );
    
    expect(container.firstChild).toBeInTheDocument();
  });
});
