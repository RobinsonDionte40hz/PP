# Workflow Testing Guide

This document outlines all user workflows that should be tested before release.

## Test Environment Setup

### Prerequisites
- Docker and Docker Compose installed
- `.env` file configured
- All containers running (backend, frontend, worker, redis, postgres)

### Start Test Environment
```bash
# Development mode
docker compose up -d

# Production mode
docker compose -f docker-compose.prod.yml up -d

# Check all services are healthy
docker compose ps
```

## Test Workflow 1: Simple Prediction Submission

**Objective**: User submits a protein prediction and monitors it to completion.

**Steps**:
1. Navigate to Dashboard (http://localhost:3000)
   - ✅ Dashboard loads without errors
   - ✅ System status shows all services online
   - ✅ Quick actions card is visible

2. Click "New Prediction" or navigate to Prediction Form
   - ✅ Stepper interface loads with 3 steps
   - ✅ Step 1: Sequence Input is active

3. Enter protein sequence
   - ✅ Enter test sequence: `ACDEFGHIKLMNPQRSTVWY`
   - ✅ Validation shows sequence is valid (20 amino acids)
   - ✅ Can click "Next" to proceed

4. Configure prediction (Step 2)
   - ✅ Preset selection shows 3 options (Fast, Balanced, High Accuracy)
   - ✅ Select "Fast" preset
   - ✅ Configuration parameters auto-fill
   - ✅ Can expand "Advanced Options"
   - ✅ Can modify iterations (set to 100 for testing)
   - ✅ Can modify agents (set to 3 for testing)

5. Review and submit (Step 3)
   - ✅ Review shows all entered information
   - ✅ Estimated time is calculated
   - ✅ Click "Submit Prediction"
   - ✅ Success message appears
   - ✅ Redirects to Live Monitoring page

6. Monitor prediction
   - ✅ Progress bar shows percentage
   - ✅ Metrics cards update in real-time
   - ✅ Energy chart shows decreasing trend
   - ✅ RMSD chart shows values
   - ✅ Event log shows progress messages
   - ✅ Agent count displays correctly
   - ✅ Pause button is functional
   - ✅ Stop button is functional

7. Wait for completion
   - ✅ Progress reaches 100%
   - ✅ Status changes to "Completed"
   - ✅ "View Results" button appears
   - ✅ All final metrics are displayed

8. View results
   - ✅ Click "View Results"
   - ✅ Redirects to Results Analysis page
   - ✅ Summary tab shows overview
   - ✅ Quality badge shows appropriate rating
   - ✅ Energy breakdown chart loads
   - ✅ Can switch to other tabs (Detailed Metrics, Trajectory, Geometric Analysis)

**Expected Duration**: 2-5 minutes for 100 iterations with 3 agents

**Success Criteria**: 
- Prediction completes without errors
- Final RMSD < 15.0 Å
- Final energy is negative (< 0 kcal/mol)
- All visualizations load correctly

---

## Test Workflow 2: Campaign Creation and Management

**Objective**: User creates a multi-protein campaign and monitors progress.

**Steps**:
1. Navigate to Campaign Management
   - ✅ Campaign list page loads
   - ✅ "New Campaign" button is visible

2. Create new campaign
   - ✅ Click "New Campaign"
   - ✅ Campaign creation dialog opens
   - ✅ Enter campaign name: "Test Campaign"
   - ✅ Select proteins: 1UBQ, 1CRN (from dropdown or text input)
   - ✅ Set phases: 2
   - ✅ Set iterations per phase: 50
   - ✅ Click "Create Campaign"
   - ✅ Campaign appears in list

3. Monitor campaign
   - ✅ Click on campaign to view details
   - ✅ Phase progress shows current phase
   - ✅ Protein results table shows each protein's status
   - ✅ Can pause/resume campaign
   - ✅ Statistics chart updates with results

4. View phase details
   - ✅ Click on individual phase
   - ✅ Phase-specific metrics display
   - ✅ Quality gates are shown
   - ✅ Can see which proteins passed/failed

5. Wait for completion
   - ✅ All proteins complete
   - ✅ Campaign status changes to "Completed"
   - ✅ Final statistics are calculated

6. Export results
   - ✅ Click "Export" button
   - ✅ Choose format (JSON/CSV)
   - ✅ File downloads successfully
   - ✅ File contains all campaign data

**Expected Duration**: 5-10 minutes depending on protein sizes

**Success Criteria**:
- Campaign completes all phases
- At least 50% of proteins meet quality gates
- Statistical analysis is generated
- Export files are valid

---

## Test Workflow 3: History Browsing and Comparison

**Objective**: User browses past predictions and compares results.

**Steps**:
1. Navigate to History Browser
   - ✅ History page loads with all past predictions
   - ✅ Can toggle between card and table view
   - ✅ Pagination works correctly

2. Use filters
   - ✅ Filter by status (Completed, Running, Failed)
   - ✅ Filter by quality (Excellent, Good, etc.)
   - ✅ Filter by date range
   - ✅ Search by protein sequence
   - ✅ Results update correctly

3. Sort results
   - ✅ Sort by date (newest/oldest)
   - ✅ Sort by RMSD (lowest/highest)
   - ✅ Sort by energy (lowest/highest)
   - ✅ Sorting works in both views

4. Select predictions for comparison
   - ✅ Select 2-3 predictions using checkboxes
   - ✅ "Compare" button becomes enabled
   - ✅ Click "Compare"
   - ✅ Comparison modal opens

5. View comparison
   - ✅ Side-by-side metrics comparison shows
   - ✅ Overlay charts display both trajectories
   - ✅ Difference calculations are shown
   - ✅ Can export comparison data

6. View individual result
   - ✅ Click on any prediction card
   - ✅ Navigates to detailed results page
   - ✅ All tabs load correctly

**Expected Duration**: 2-3 minutes

**Success Criteria**:
- All filters work correctly
- Comparison shows meaningful differences
- No data inconsistencies

---

## Test Workflow 4: 3D Structure Visualization

**Objective**: User visualizes predicted protein structure in 3D.

**Steps**:
1. From results page, click "3D Visualization"
   - ✅ Structure Visualization page loads
   - ✅ NGL Viewer initializes
   - ✅ Protein structure renders

2. Test representation controls
   - ✅ Switch to Cartoon representation
   - ✅ Switch to Backbone representation
   - ✅ Switch to Ball+Stick representation
   - ✅ Rendering updates smoothly

3. Test coloring schemes
   - ✅ Apply Secondary Structure coloring
   - ✅ Apply Residue Type coloring
   - ✅ Apply B-factor coloring
   - ✅ Colors apply correctly

4. Interact with structure
   - ✅ Rotate structure with mouse
   - ✅ Zoom in/out with scroll
   - ✅ Pan with right-click drag
   - ✅ Smooth 60fps interaction

5. Use controls
   - ✅ Reset view button works
   - ✅ Center structure button works
   - ✅ Toggle fullscreen works
   - ✅ Spin animation toggles

6. Highlight geometric patterns
   - ✅ Enable geometric highlighting
   - ✅ Icosahedron patterns show in color
   - ✅ Dodecahedron patterns show
   - ✅ Can toggle individual patterns

7. Export structure
   - ✅ Click "Export PDB"
   - ✅ PDB file downloads
   - ✅ File is valid PDB format
   - ✅ Take screenshot
   - ✅ PNG file downloads

**Expected Duration**: 3-5 minutes

**Success Criteria**:
- Structure renders without errors
- All representations work
- Interactions are smooth (60fps)
- Export files are valid

---

## Test Workflow 5: Settings Configuration

**Objective**: User configures system settings and preferences.

**Steps**:
1. Navigate to Settings page
   - ✅ Settings page loads with tabs
   - ✅ System Configuration tab is active

2. Modify system settings
   - ✅ Change default iterations
   - ✅ Change default agents
   - ✅ Select different preset as default
   - ✅ Changes are reflected immediately

3. Configure visualization settings
   - ✅ Switch to Visualization tab
   - ✅ Change default representation
   - ✅ Change default coloring scheme
   - ✅ Toggle geometric highlighting

4. Configure notifications
   - ✅ Switch to Notifications tab
   - ✅ Enable/disable prediction completion alerts
   - ✅ Enable/disable error notifications
   - ✅ Set notification preferences

5. Advanced settings
   - ✅ Switch to Advanced tab
   - ✅ View system information
   - ✅ Clear cache
   - ✅ Export settings
   - ✅ Import settings

6. Save and apply
   - ✅ Click "Save Changes"
   - ✅ Success message appears
   - ✅ Settings persist after page reload

7. Reset to defaults
   - ✅ Click "Reset to Defaults"
   - ✅ Confirmation dialog appears
   - ✅ Settings reset correctly

**Expected Duration**: 2-3 minutes

**Success Criteria**:
- All settings save correctly
- Changes persist across sessions
- Reset functionality works
- No data loss

---

## Test Workflow 6: Error Handling and Recovery

**Objective**: System handles errors gracefully and allows recovery.

**Steps**:
1. Test invalid sequence input
   - ✅ Enter invalid characters in sequence
   - ✅ Validation error displays
   - ✅ Error message is clear and helpful
   - ✅ Cannot proceed to next step

2. Test network disconnection
   - ✅ Disconnect network during prediction
   - ✅ WebSocket disconnection detected
   - ✅ Error message appears
   - ✅ Reconnect network
   - ✅ System auto-reconnects
   - ✅ Prediction continues

3. Test backend unavailable
   - ✅ Stop backend container
   - ✅ API calls show error
   - ✅ User-friendly error message displays
   - ✅ Restart backend
   - ✅ System recovers

4. Test pause and resume
   - ✅ Start prediction
   - ✅ Click pause during execution
   - ✅ Prediction pauses successfully
   - ✅ Click resume
   - ✅ Prediction continues from same point
   - ✅ No data loss

5. Test checkpoint recovery
   - ✅ Start long prediction
   - ✅ Stop after 50 iterations
   - ✅ Restart and resume from checkpoint
   - ✅ Prediction continues correctly
   - ✅ Metrics match previous session

**Expected Duration**: 5-10 minutes

**Success Criteria**:
- All errors display user-friendly messages
- System recovers from temporary failures
- No data corruption
- Checkpoints work correctly

---

## Test Workflow 7: Real PP System Integration

**Objective**: Test with actual UBF protein system predictions.

**Steps**:
1. Submit real prediction through UI
   - ✅ Use realistic sequence (20-50 amino acids)
   - ✅ Set iterations to 500-1000
   - ✅ Set agents to 5-10
   - ✅ Use "Balanced" preset

2. Verify backend integration
   - ✅ Check backend logs for UBF invocation
   - ✅ Verify checkpoint files are created
   - ✅ Verify visualization output is generated
   - ✅ Verify PDB cache is populated

3. Monitor progress
   - ✅ WebSocket updates arrive regularly
   - ✅ Metrics match UBF output
   - ✅ Event log shows UBF events
   - ✅ Agent status updates correctly

4. Verify results parsing
   - ✅ Final results match UBF output files
   - ✅ RMSD values are correct
   - ✅ Energy values are correct
   - ✅ Trajectory data is complete

5. Check file system
   - ✅ Results directory has output files
   - ✅ Checkpoints directory has checkpoints
   - ✅ PDB cache has structure files
   - ✅ Visualization output has charts

6. Test with QCPP integration
   - ✅ Enable QCPP integration in config
   - ✅ Submit prediction
   - ✅ Verify QCPP analysis runs
   - ✅ Verify quantum metrics appear in results

**Expected Duration**: 10-30 minutes depending on configuration

**Success Criteria**:
- Integration works end-to-end
- All output files are created
- Results match command-line execution
- No integration errors

---

## Performance Benchmarks

### Response Time Targets
- Health check: < 100ms
- Prediction submission: < 500ms
- Status query: < 200ms
- Results retrieval: < 1s
- WebSocket message: < 1s latency

### Load Testing
- Concurrent predictions: 5+ without degradation
- Concurrent users: 10+ monitoring same prediction
- Large sequences: 200+ amino acids handled
- Long campaigns: 10+ proteins per campaign

### Frontend Performance
- Initial page load: < 3s
- Route transition: < 500ms
- Chart rendering: < 2s for 1000 points
- 3D visualization: 60fps interaction

---

## Automated Test Execution

### Run E2E Tests
```bash
# Start test environment
docker compose up -d

# Wait for services to be ready
timeout /t 30

# Run automated E2E tests
python tests\e2e_test.py --url http://localhost:8000

# Check results
type tests\e2e_test_report.json
```

### Run Docker Deployment Test
```bash
# Clean environment
docker compose down -v

# Run deployment test
tests\test_docker_deployment.bat
```

### Run Backend Tests
```bash
cd backend
pytest tests/ -v --cov=app --cov-report=html
```

### Run Frontend Tests
```bash
cd frontend
npm test
npm run test:coverage
```

---

## Bug Reporting Template

When bugs are found during testing, use this template:

```markdown
### Bug Report

**Title**: [Brief description]

**Severity**: Critical / High / Medium / Low

**Workflow**: [Which workflow was being tested]

**Steps to Reproduce**:
1. 
2. 
3. 

**Expected Behavior**:

**Actual Behavior**:

**Error Messages**:
```
[paste error messages or logs]
```

**Environment**:
- Browser: 
- Docker version: 
- OS: 

**Screenshots**: [if applicable]

**Logs**: [attach relevant logs]
```

---

## Test Sign-Off Checklist

Before declaring Task 7.4 complete, ensure:

- [ ] All 7 workflows tested and passed
- [ ] All automated tests passing
- [ ] Docker deployment works on clean system
- [ ] Performance benchmarks met
- [ ] All critical bugs fixed
- [ ] All high-priority bugs fixed
- [ ] Documentation verified and accurate
- [ ] Release notes created
- [ ] Test report generated

**Tested by**: _________________
**Date**: _________________
**Sign-off**: _________________
