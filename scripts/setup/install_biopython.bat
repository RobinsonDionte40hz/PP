@echo off
echo Setting up Visual Studio Build Environment...
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

echo Installing all QCPP dependencies...
myvenv\Scripts\python.exe -m pip install -r requirements_qcpp.txt

echo Verifying installation...
myvenv\Scripts\python.exe -c "import numpy, scipy, matplotlib, pandas, sklearn, statsmodels; from Bio import PDB; print('All packages imported successfully!')"

echo Done!
pause
