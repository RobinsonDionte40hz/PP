@echo off
REM Quick validation test suite for geometric hypothesis
REM Windows batch script

echo ================================================================================
echo GEOMETRIC HYPOTHESIS VALIDATION - QUICK TEST SUITE
echo ================================================================================
echo.

REM Test 1: Small ordered protein (should show high phi if hypothesis correct)
echo [TEST 1/4] Small ordered protein: 1VII (36 residues)
echo --------------------------------------------------------------------------------
python quick_validation_test.py --pdb 1VII --iterations 300 --agents 8 --output validation_1VII.json
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Test 1 failed!
    pause
    exit /b 1
)
echo.
echo.

REM Test 2: Medium ordered protein  
echo [TEST 2/4] Medium ordered protein: 1UBQ (76 residues) 
echo --------------------------------------------------------------------------------
python quick_validation_test.py --pdb 1UBQ --iterations 400 --agents 10 --output validation_1UBQ.json
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Test 2 failed!
    pause
    exit /b 1
)
echo.
echo.

REM Test 3: IDP (should show low phi if hypothesis correct)
echo [TEST 3/4] Intrinsically disordered protein: 1CD3 (143 residues)
echo --------------------------------------------------------------------------------  
python quick_validation_test.py --pdb 1CD3 --iterations 400 --agents 10 --output validation_1CD3.json
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Test 3 failed!
    pause
    exit /b 1
)
echo.
echo.

REM Test 4: Another IDP for comparison
echo [TEST 4/4] IDP alpha-synuclein: 1MVF (127 residues)
echo --------------------------------------------------------------------------------
python quick_validation_test.py --pdb 1MVF --iterations 400 --agents 10 --output validation_1MVF.json
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Test 4 failed!
    pause
    exit /b 1
)
echo.
echo.

echo ================================================================================
echo TESTS COMPLETE - ANALYZING RESULTS
echo ================================================================================
echo.

REM Create summary report
python -c "import json; import glob; results = {}; [results.update({f.split('_')[1].split('.')[0]: json.load(open(f))}) for f in glob.glob('validation_*.json')]; print('\n=== SUMMARY COMPARISON ===\n'); print('Protein | Type | Native φ | Predicted φ | Δφ | RMSD | Quality'); print('-'*75); [print(f'{pid:<7} | {\"IDP\" if pid in [\"1CD3\",\"1MVF\"] else \"ORD\":<4} | {r[\"native\"][\"phi_percentage\"]:>8.2f}%% | {r[\"predicted\"][\"phi_percentage\"]:>11.2f}%% | {r[\"predicted\"][\"phi_percentage\"]-r[\"native\"][\"phi_percentage\"]:>+6.2f}%% | {r.get(\"rmsd_true\",0):>5.2f}Å | {r.get(\"quality\",\"N/A\")}') for pid, r in results.items()]; print(); print('KEY FINDINGS:'); print('-'*75); ordered_phi = [r['predicted']['phi_percentage'] for pid, r in results.items() if pid not in ['1CD3','1MVF']]; idp_phi = [r['predicted']['phi_percentage'] for pid, r in results.items() if pid in ['1CD3','1MVF']]; print(f'Ordered proteins: φ = {sum(ordered_phi)/len(ordered_phi):.2f}%% (n={len(ordered_phi)})'); print(f'IDP proteins: φ = {sum(idp_phi)/len(idp_phi):.2f}%% (n={len(idp_phi)})'); diff = sum(ordered_phi)/len(ordered_phi) - sum(idp_phi)/len(idp_phi); print(f'Difference: Δφ = {diff:+.2f}%%'); print(); if abs(diff) < 3: print('⚠️  HYPOTHESIS CHALLENGED: IDPs show similar φ to ordered proteins!'); print('    → Geometric patterns likely reflect algorithm bias'); else: print('✓ HYPOTHESIS SUPPORTED: IDPs show lower φ than ordered proteins'); print('    → Geometric patterns may reflect real folding principles')"

echo.
echo ================================================================================
echo Results saved to validation_*.json files
echo See VALIDATION_GUIDE.md for interpretation
echo ================================================================================
echo.

pause
