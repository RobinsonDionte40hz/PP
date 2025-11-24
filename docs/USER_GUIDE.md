# User Guide - Protein Prediction Platform

Welcome to the Protein Prediction Platform user guide! This document will walk you through all features of the web interface.

## Table of Contents

- [Getting Started](#getting-started)
  - [Account Access](#account-access)
  - [First Time Setup](#first-time-setup)
- [Dashboard](#dashboard)
- [Creating Predictions](#creating-predictions)
- [Monitoring Predictions](#monitoring-predictions)
- [Analyzing Results](#analyzing-results)
- [3D Structure Visualization](#3d-structure-visualization)
- [Campaign Management](#campaign-management)
- [History Browser](#history-browser)
- [Settings](#settings)

## Getting Started

### Account Access

The platform requires authentication for all features.

#### Login

1. **Navigate to**: http://localhost:3000
2. **Enter credentials**: Username and password
3. **Click Login**: You'll be redirected to the dashboard

#### Master Test Accounts (Development)

For testing and development, use these pre-configured accounts:

- **Admin**: username=`admin`, password=`Admin@2025!`
  - Full system access
  - User management capabilities
  - All prediction features

- **Developer**: username=`developer`, password=`Dev@2025!`
  - Enhanced debugging access
  - All prediction features
  - Performance monitoring tools

#### New User Registration

1. Click "Register here" on the login page
2. Fill in:
   - Username (3-50 characters)
   - Email address (valid format)
   - Password (min 8 characters, includes uppercase, lowercase, number, special character)
3. Click "Register"
4. Automatically logged in and redirected to dashboard

### First Time Setup

1. **Access the application**: Open your browser and navigate to http://localhost:3000
2. **Log in**: Use your credentials or master test account
3. **Explore the dashboard**: You'll see the main dashboard with quick actions and system status
4. **Try a quick prediction**: Click "New Prediction" to submit your first prediction

### Interface Overview

The interface consists of:

- **Header**: System status, theme toggle, and navigation
- **Sidebar**: Main navigation menu
- **Content Area**: Current page content
- **Footer**: Version information and links

### Navigation

- **Dashboard**: Overview and quick actions
- **New Prediction**: Submit a new prediction
- **Live Monitoring**: Monitor running predictions in real-time
- **Results**: View and analyze completed predictions
- **3D Viewer**: Visualize protein structures
- **Campaigns**: Manage systematic testing campaigns
- **History**: Browse all past predictions
- **Settings**: Configure application settings

## Dashboard

The dashboard provides an at-a-glance view of your protein prediction activities.

### Quick Actions

- **New Prediction**: Submit a single prediction
- **New Campaign**: Start a systematic campaign
- **View History**: Browse past predictions
- **Settings**: Configure preferences

### System Status

Shows real-time status of:
- **Backend API**: Connection status
- **Redis**: Queue status
- **Celery Workers**: Number of active workers
- **Active Predictions**: Currently running predictions

### Recent Predictions

Displays your 5 most recent predictions with:
- Sequence name or ID
- Status badge
- Quality score
- Creation time
- Quick actions (view, monitor, delete)

### Statistics

Shows aggregate metrics:
- **Total Predictions**: Lifetime count
- **Success Rate**: Percentage of successful predictions
- **Average RMSD**: Mean RMSD across all predictions
- **Average Energy**: Mean energy across all predictions

## Creating Predictions

### Step 1: Sequence Input

Navigate to **New Prediction** from the sidebar or dashboard.

#### Input Methods

**1. Manual Entry**
- Type or paste your protein sequence (1-letter amino acid codes)
- Supports standard 20 amino acids: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
- Example: `MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG`

**2. FASTA File Upload**
- Click "Upload FASTA" button
- Select a `.fasta` or `.fa` file
- System extracts sequence automatically

**3. Example Sequences**
- Click "Load Example" dropdown
- Choose from predefined proteins:
  - Ubiquitin (1UBQ) - 76 residues
  - Crambin (1CRN) - 46 residues
  - Small peptide (2MR9) - 44 residues

#### Sequence Validation

The system validates:
- ✅ Only valid amino acid codes
- ✅ Minimum length (typically 10 residues)
- ✅ Maximum length (typically 1000 residues)
- ❌ Invalid characters highlighted in red

### Step 2: Configuration

Configure prediction parameters:

#### Basic Settings

**Iterations** (100-10000, default: 1000)
- Number of optimization cycles
- More iterations = better results but longer runtime
- Recommended: 1000 for quick tests, 5000 for production

**Agents** (1-100, default: 10)
- Number of exploration agents
- More agents = diverse exploration but more memory
- Recommended: 10 for balanced, 50 for thorough

**Diversity Strategy**
- **Cautious**: Conservative exploration (33% of agents)
- **Balanced**: Mixed strategies (34% of agents)
- **Aggressive**: Bold exploration (33% of agents)
- **Mixed**: Combination of all (default)

#### Advanced Features

**Enable QCPP Integration** ✅ (Recommended)
- Integrates quantum coherence physics calculations
- Provides real-time quantum feedback
- Improves structural accuracy

**Enable Mediator Agents**
- Adds specialized pattern detection agents
- Helps coordinate exploration
- Useful for complex proteins

**Enable Geometric Targeting**
- Targets golden ratio geometric patterns
- Guides toward stable structures
- Experimental feature

**Enable Quantum Refinement**
- Two-stage optimization with quantum physics
- Refines structures after initial fold
- Best results but slower

#### QCPP Configuration Preset

- **Default**: Balanced speed and accuracy
- **High Performance**: Faster, less frequent analysis
- **High Accuracy**: Slower, more frequent analysis

### Step 3: Review & Submit

Review your configuration:
- Sequence summary
- All selected parameters
- Estimated runtime
- Resource requirements

Click **Submit Prediction** to queue the job.

You'll be redirected to the Live Monitoring page automatically.

## Monitoring Predictions

The Live Monitoring page provides real-time updates during prediction execution.

### Header Section

- **Prediction ID**: Unique identifier
- **Progress Bar**: Visual progress indicator
- **Status Badge**: Current status (Running, Paused, etc.)
- **Controls**: Pause/Resume/Stop buttons

### Metrics Grid

Six key metrics displayed in cards:

**Current Energy** (kcal/mol)
- Lower is better
- Negative values indicate folded structures
- Target: < -100 kcal/mol for small proteins

**Current RMSD** (Ångströms)
- Lower is better
- Measures deviation from native structure
- Target: < 5Å for good quality

**Best Energy**
- Lowest energy achieved so far
- Indicates best structure found

**Best RMSD**
- Lowest RMSD achieved
- Best structural accuracy

**Iteration Progress**
- Current / Total iterations
- Percentage complete

**Estimated Time**
- Time remaining until completion
- Updates dynamically based on speed

### Live Charts

Three interactive charts update in real-time:

**Energy Trajectory**
- X-axis: Iteration number
- Y-axis: Energy (kcal/mol)
- Shows energy minimization progress
- Hover for exact values

**RMSD Progress**
- X-axis: Iteration number
- Y-axis: RMSD (Å)
- Shows structural accuracy improvement
- Look for decreasing trend

**Exploration Parameters**
- X-axis: Iteration number
- Y-axis: Parameter values
- Shows aggressiveness and consistency
- Indicates exploration behavior

### Event Log

Real-time event stream showing:

- **Milestones**: New best energy/RMSD
- **Warnings**: Stuck detection, high energy
- **Info**: Phase transitions, checkpoints
- **Errors**: Calculation failures

**Filtering**:
- All Events
- Milestones Only
- Warnings Only
- Errors Only

**Actions**:
- Auto-scroll toggle
- Clear log
- Export log to file

### Structure Preview

Click "View 3D Structure" to open a modal with:
- Current predicted structure (PDB format)
- Interactive 3D visualization
- Download current structure button

### Controls

**Pause**: Temporarily halt execution
- Progress saved automatically
- Resume anytime
- Useful for system maintenance

**Resume**: Continue paused prediction
- Picks up where it left off
- No data loss

**Stop**: Permanently end prediction
- Cannot be resumed
- Results saved up to stop point
- Use if prediction is not progressing

## Analyzing Results

After completion, navigate to the Results Analysis page.

### Summary Tab

Overview of the prediction:

**Quality Metrics**
- **Final Energy**: Total energy of best structure
- **Final RMSD**: Accuracy vs native structure
- **Quality Score**: Combined metric (0-1, higher is better)
- **Structure Type**: Secondary structure composition

**Performance**
- Total iterations run
- Duration (HH:MM:SS)
- Iterations per second
- Memory used

**Quality Badge**
- Excellent: RMSD < 2Å, Energy < -200
- Good: RMSD 2-4Å, Energy -100 to -200
- Acceptable: RMSD 4-5Å, Energy -50 to -100
- Poor: RMSD > 5Å, Energy > -50

**Actions**
- Download PDB structure
- Download JSON results
- Export PDF report
- Compare with other predictions
- View in 3D Viewer

### Detailed Metrics Tab

Comprehensive metrics display:

**Energy Breakdown** (Pie Chart)
- Bond energy
- Angle energy
- Dihedral energy
- Van der Waals
- Electrostatic
- Hydrogen bonds

**Secondary Structure Distribution** (Bar Chart)
- Helix content (%)
- Sheet content (%)
- Coil content (%)

**Agent Statistics** (Table)
Per-agent metrics:
- Agent ID
- Total moves attempted
- Moves accepted
- Acceptance rate
- Final energy
- Final parameters

**Memory Usage**
- Memories stored
- Memories shared
- Average significance
- Top significant events

### Trajectory Tab

Visualize the optimization path:

**Interactive Charts**
- Energy vs Iteration (line chart)
- RMSD vs Iteration (line chart)
- Parameters vs Iteration (multi-line chart)

**Statistics**
- Best iteration number
- Improvement rate
- Convergence point
- Stability indicators

**Export Options**
- Download trajectory data (CSV)
- Export charts as images

### Geometric Analysis Tab

If geometric targeting was enabled:

**Patterns Detected**
- Icosahedron features
- Dodecahedron features
- Octahedron features
- Golden spiral patterns

**Phi Ratio Analysis**
- Phi-based distances count
- Golden ratio score (0-1)
- Harmonic resonance

**Geometric Relationships**
- Distance distributions
- Angle distributions
- Pattern significance

## 3D Structure Visualization

The Structure Visualization page provides advanced 3D viewing capabilities.

### Loading a Structure

**From Results**: Click "View in 3D Viewer" from results page

**From History**: Click 3D icon in history browser

**Manual Upload**: Upload PDB file directly

### Viewer Controls

**Representation**
- **Cartoon**: Alpha helix and beta sheet ribbons (default)
- **Backbone**: C-alpha trace
- **Ball & Stick**: Atomic detail
- **Surface**: Molecular surface
- **Ribbon**: Simple ribbon

**Color Scheme**
- **Secondary Structure**: Helix (red), Sheet (yellow), Coil (white)
- **Residue**: By residue type
- **Hydrophobicity**: Hydrophobic (orange) to hydrophilic (blue)
- **B-factor**: By flexibility
- **Chain**: By chain ID

**Background**
- White (default)
- Black
- Gradient

### Comparison Mode

Compare predicted vs native structures:

1. Load predicted structure
2. Click "Load Native for Comparison"
3. Enter PDB ID (e.g., 1UBQ)
4. System fetches and overlays native structure
5. Toggle visibility of each structure
6. View RMSD alignment

### Interactions

**Mouse Controls**
- **Left-click drag**: Rotate structure
- **Right-click drag**: Pan
- **Scroll**: Zoom in/out
- **Double-click**: Center on atom/residue

**Keyboard Shortcuts**
- `R`: Reset view
- `F`: Full screen
- `S`: Screenshot
- `Space`: Toggle spin

### Actions

**Screenshot**: Capture current view as PNG

**Export PDB**: Download structure file

**Geometric Highlights**: Show detected patterns
- Golden ratio distances highlighted
- Phi-based contacts shown
- Harmonic clusters colored

**Sequence Viewer**: Show residue sequence with selection

## Campaign Management

For systematic testing of multiple proteins with various configurations.

### Creating a Campaign

1. Navigate to **Campaigns** → **Create New Campaign**
2. Enter campaign name
3. Add proteins (PDB IDs or custom sequences)
4. Define test configurations:
   - Name each configuration
   - Set parameters (iterations, agents, features)
   - Add quality gates (optional)
5. Review and submit

### Campaign Progress

The campaign detail view shows:

**Phase Progress**
- Circular progress indicator per phase
- Phase number and name
- Status: pending/running/completed/failed
- Predictions completed / total

**Overall Statistics**
- Total proteins tested
- Predictions completed
- Success rate
- Average metrics

**Current Activity**
- Currently running protein
- Current configuration
- Estimated completion

### Protein Results Table

Table of all predictions in the campaign:

| Protein | Config | Status | RMSD | Energy | Duration | Actions |
|---------|--------|--------|------|--------|----------|---------|
| 1UBQ | Base | ✅ | 7.2Å | -189 | 15:23 | View |
| 1UBQ | High Iter | ⏳ | - | - | - | Monitor |

**Sorting**: Click column headers to sort

**Filtering**: Filter by status, protein, configuration

**Actions**: View results, monitor live, download, delete

### Campaign Statistics

Charts and analyses:

**Performance by Protein** (Bar Chart)
- Average RMSD per protein
- Average energy per protein

**Performance by Configuration** (Bar Chart)
- Average RMSD per config
- Average energy per config

**Success Rate** (Pie Chart)
- Completed successfully
- Failed
- In progress

**Duration Distribution** (Histogram)
- Time taken per prediction
- Identify outliers

### Failure Analysis

If predictions fail:
- View error messages
- See failure patterns
- Retry failed predictions
- Adjust configurations

## History Browser

Browse all your past predictions with powerful search and filtering.

### View Modes

**Card View**: Visual cards with key metrics

**Table View**: Compact table format

Toggle between views with buttons at top-right.

### Search & Filter

**Search Bar**: Search by:
- Sequence content
- Prediction ID
- Date range

**Status Filter**
- All
- Completed
- Failed
- Running

**Quality Filter**
- All
- Excellent
- Good
- Acceptable
- Poor

**Date Range**
- Last 24 hours
- Last week
- Last month
- Custom range

**Advanced Filters** (click "Advanced")
- RMSD range
- Energy range
- Duration range
- Features used (QCPP, mediators, etc.)

### Sorting

Sort by:
- Created date (newest/oldest)
- RMSD (best/worst)
- Energy (best/worst)
- Duration (fastest/slowest)

### Comparison Tool

1. Select multiple predictions (checkboxes)
2. Click "Compare Selected" button
3. View side-by-side comparison:
   - Metrics table
   - Chart overlays
   - Best performer highlights
4. Export comparison report

### Bulk Actions

Select multiple predictions:
- Delete selected
- Export selected
- Add to campaign

## Settings

Customize application behavior and appearance.

### System Configuration Tab

**Default Prediction Settings**
- Default iterations
- Default agents
- Default features enabled

**Performance**
- Max concurrent predictions
- Auto-save interval
- Cache size

**File Management**
- Auto-delete old results (days)
- Max storage per user
- PDB cache settings

### Visualization Tab

**3D Viewer Defaults**
- Default representation
- Default color scheme
- Default background
- Auto-spin enabled

**Chart Settings**
- Chart theme (light/dark)
- Default chart type
- Animation speed

**Data Display**
- Decimal places for metrics
- Number format (scientific/decimal)
- Date/time format

### Notifications Tab

**Real-time Notifications**
- Browser notifications
- Sound alerts
- Desktop notifications (requires permission)

**Email Notifications** (if configured)
- Prediction completed
- Prediction failed
- Campaign completed

**Notification Events**
- Completion
- Milestones (new best RMSD/energy)
- Warnings (stuck, high energy)
- Errors

### Advanced Tab

**System Information**
- Backend version
- Frontend version
- API status
- Redis status
- Worker count

**Data Management**
- Export all results
- Import results
- Clear cache
- Reset settings

**Developer Options**
- Enable debug mode
- Show performance metrics
- API request logging

## Tips & Best Practices

### For Best Results

1. **Start small**: Test with short sequences (< 100 residues) first
2. **Use QCPP**: Enable QCPP integration for better accuracy
3. **Monitor progress**: Watch energy/RMSD trends for convergence
4. **Compare configurations**: Run multiple configs to find optimal settings
5. **Enable refinement**: For final predictions, use quantum refinement

### Performance

1. **Batch processing**: Use campaigns for multiple predictions
2. **Adjust iterations**: 1000 for quick tests, 5000 for production
3. **Resource management**: Don't run too many simultaneous predictions
4. **Clean up**: Delete old results periodically

### Troubleshooting

1. **Stuck predictions**: Use stop/resume or adjust parameters
2. **High RMSD**: Try more iterations, enable refinement
3. **Slow performance**: Reduce agents or iterations
4. **Connection lost**: Check system status on dashboard

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `N` | New Prediction |
| `D` | Go to Dashboard |
| `H` | Go to History |
| `M` | Go to Monitoring (if prediction running) |
| `S` | Go to Settings |
| `?` | Show help |
| `/` | Focus search |
| `Esc` | Close modal/dialog |

## Support

Need help? Check these resources:

- [Setup Guide](SETUP.md) - Installation help
- [API Documentation](API.md) - API reference
- [Developer Guide](DEVELOPER_GUIDE.md) - Technical details
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues

Or create an issue on GitHub.
