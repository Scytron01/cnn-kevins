@echo off
setlocal
cd /d "%~dp0\.."
where vsim >nul 2>nul
if errorlevel 1 (
  echo ERROR: vsim.exe is not in PATH. Open the ModelSim-Altera command prompt first.
  exit /b 1
)
vsim -c -do verilog/run_cnn.do
endlocal
