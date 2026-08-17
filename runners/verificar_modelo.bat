@echo off
echo Verificando modelo ibm/granite4.1:3b-q6_K...
echo.
ollama list | findstr granite4.1:3b-q6
if %ERRORLEVEL% EQU 0 (
    echo.
    echo OK: Modelo encontrado
) else (
    echo.
    echo ERROR: Modelo no encontrado
)
