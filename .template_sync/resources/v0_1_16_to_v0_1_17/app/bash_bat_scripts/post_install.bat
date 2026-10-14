@ECHO OFF
SETLOCAL EnableExtensions DisableDelayedExpansion

SET "LOG_FILE=%PREFIX%\menuinst_debug.log"
SET "PYTHON_EXE=%PREFIX%\python.exe"
SET "PROJECT_ROOT=%PREFIX%\PROJECT_NAME"
SET "BASE_REQUIREMENTS=%PROJECT_ROOT%\requirements.txt"
SET "GPU_REQUIREMENTS=%PROJECT_ROOT%\requirements_gpu.txt"
SET "SELECTED_REQUIREMENTS=%BASE_REQUIREMENTS%"
SET "NVIDIA_SMI="
SET "CA_BUNDLE="
SET "FAILURE_MESSAGE="
SET "CERTIFI_PATH_FILE=%TEMP%\labconstrictor_certifi_%RANDOM%_%RANDOM%.txt"

> "%LOG_FILE%" echo Running post_install

IF NOT EXIST "%PYTHON_EXE%" (
    SET "FAILURE_MESSAGE=Bundled Python executable was not found at %PYTHON_EXE%."
    GOTO :fail
)
IF NOT EXIST "%PROJECT_ROOT%\launch_jupyter.py" (
    SET "FAILURE_MESSAGE=TLS-resilient launcher was not found at %PROJECT_ROOT%\launch_jupyter.py."
    GOTO :fail
)
IF NOT EXIST "%BASE_REQUIREMENTS%" (
    SET "FAILURE_MESSAGE=Base requirements file was not found at %BASE_REQUIREMENTS%."
    GOTO :fail
)

REM pip 24.2+ uses the operating-system trust store by default. Some Windows
REM certificate stores contain malformed entries that OpenSSL cannot parse.
REM Keep verification enabled while opting out of system trust for installer
REM pip operations and using an explicit verified CA bundle.
SET "PIP_USE_DEPRECATED=legacy-certs"

"%PYTHON_EXE%" "%PROJECT_ROOT%\launch_jupyter.py" --print-ca-bundle > "%CERTIFI_PATH_FILE%" 2>> "%LOG_FILE%"
IF ERRORLEVEL 1 (
    SET "FAILURE_MESSAGE=Unable to locate a verified CA bundle."
    GOTO :fail
)
SET /P "CA_BUNDLE="<"%CERTIFI_PATH_FILE%"
DEL /Q "%CERTIFI_PATH_FILE%" >NUL 2>&1

IF NOT DEFINED CA_BUNDLE (
    SET "FAILURE_MESSAGE=No verified CA bundle could be selected."
    GOTO :fail
)
IF NOT EXIST "%CA_BUNDLE%" (
    SET "FAILURE_MESSAGE=Selected CA bundle does not exist: %CA_BUNDLE%"
    GOTO :fail
)

SET "PIP_CERT=%CA_BUNDLE%"
SET "REQUESTS_CA_BUNDLE=%CA_BUNDLE%"
SET "CURL_CA_BUNDLE=%CA_BUNDLE%"
echo Using verified CA bundle for installer network operations: "%CA_BUNDLE%" >> "%LOG_FILE%"

IF EXIST "%GPU_REQUIREMENTS%" (
    CALL :detect_nvidia_smi
    IF DEFINED NVIDIA_SMI (
        echo NVIDIA GPU utility detected, installing GPU requirements from "%GPU_REQUIREMENTS%" >> "%LOG_FILE%"
        SET "SELECTED_REQUIREMENTS=%GPU_REQUIREMENTS%"
    ) ELSE (
        echo NVIDIA GPU not detected, installing CPU requirements from "%BASE_REQUIREMENTS%" >> "%LOG_FILE%"
    )
) ELSE (
    echo GPU requirements file not found, installing CPU requirements from "%BASE_REQUIREMENTS%" >> "%LOG_FILE%"
)

echo Installing requirements from "%SELECTED_REQUIREMENTS%" >> "%LOG_FILE%"
"%PYTHON_EXE%" -m pip install -r "%SELECTED_REQUIREMENTS%" >> "%LOG_FILE%" 2>&1
IF ERRORLEVEL 1 (
    SET "FAILURE_MESSAGE=Application requirements installation failed."
    GOTO :fail
)

IF EXIST "%PROJECT_ROOT%\requirements-windows.txt" (
    echo Installing Windows-specific requirements. >> "%LOG_FILE%"
    "%PYTHON_EXE%" -m pip install -r "%PROJECT_ROOT%\requirements-windows.txt" >> "%LOG_FILE%" 2>&1
    IF ERRORLEVEL 1 (
        SET "FAILURE_MESSAGE=Windows-specific requirements installation failed."
        GOTO :fail
    )
)

REM External Python code is optional in LabConstrictor projects. If setup.py is
REM bundled, install it without a temporary build environment and verify that
REM the generated import package is usable. Otherwise continue without it.
IF EXIST "%PROJECT_ROOT%\setup.py" (
    echo Found setup.py, installing PROJECT_NAME package locally without build isolation. >> "%LOG_FILE%"
    "%PYTHON_EXE%" -m pip install --no-deps --no-build-isolation "%PROJECT_ROOT%" >> "%LOG_FILE%" 2>&1
    IF ERRORLEVEL 1 (
        SET "FAILURE_MESSAGE=PROJECT_NAME package installation failed."
        GOTO :fail
    )

    REM Tools first, only for an app whose package ships a tools module: the package may import labconstrictor_tools at module level,
    REM so labconstrictor-tools is installed BEFORE the import check below. A failed install is only a WARNING, see :install_lc_tools at the end of this file.
    CALL :install_lc_tools

    "%PYTHON_EXE%" -c "import PYTHON_PROJ_NAME; print('PROJECT_NAME import successful:', PYTHON_PROJ_NAME.__file__)" >> "%LOG_FILE%" 2>&1
    IF ERRORLEVEL 1 (
        SET "FAILURE_MESSAGE=Installed PROJECT_NAME package could not be imported as PYTHON_PROJ_NAME."
        GOTO :fail
    )
) ELSE (
    echo No setup.py detected; this project does not bundle an optional Python package. >> "%LOG_FILE%"
)

REM Optional: expose the app's tools to Napari, Fiji and the command line (LabConstrictor tools bridge).
REM If the app's package ships a module named <package>_lc_tools, install labconstrictor-tools and register that module.
REM This step must never fail the installation. LC_TOOLS_SPEC can point to another source (wheel, git URL, mirror).
REM labconstrictor-tools is installed by :install_lc_tools, called above before the import check of the package (it sets LC_TOOLS_PIN, LC_TOOLS_SPEC and LC_TOOLS_READY).
SET "LC_APP_VERSION="
IF EXIST "%PROJECT_ROOT%\construct.yaml" FOR /F "usebackq tokens=1,* delims=: " %%A IN (`findstr /B /C:"version:" "%PROJECT_ROOT%\construct.yaml"`) DO SET "LC_APP_VERSION=%%~B"
IF NOT DEFINED LC_APP_VERSION (
    SET "LC_APP_VERSION=0"
    echo WARNING: no version: line at the start of a line in construct.yaml - registering the tools with version 0. >> "%LOG_FILE%"
)
"%PYTHON_EXE%" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('PYTHON_PROJ_NAME_lc_tools') else 1)" >> "%LOG_FILE%" 2>&1
IF NOT ERRORLEVEL 1 (
    echo Found PYTHON_PROJ_NAME_lc_tools: registering the tools of PROJECT_NAME for Napari and Fiji. >> "%LOG_FILE%"
    REM labconstrictor-tools was installed before the import check (:install_lc_tools): register only if that worked (errorlevel 1 otherwise).
    "%PYTHON_EXE%" -c "import sys; sys.exit(0 if sys.argv[1] == '1' else 1)" "%LC_TOOLS_READY%" >> "%LOG_FILE%" 2>&1
    IF NOT ERRORLEVEL 1 "%PYTHON_EXE%" -m labconstrictor_tools register --name "PROJECT_NAME" --prefix "%PREFIX%" --module PYTHON_PROJ_NAME_lc_tools --version "%LC_APP_VERSION%" --display-name "PROJECT_NAME" >> "%LOG_FILE%" 2>&1
    IF ERRORLEVEL 1 echo WARNING: tool registration failed - see the pip and register output above in this file; PROJECT_NAME itself is installed. >> "%LOG_FILE%"
)

"%PYTHON_EXE%" "%PROJECT_ROOT%\include_path.py" --path "%PREFIX%" --files "%PROJECT_ROOT%\notebook_launcher.json" --keyword "BASE_PATH_KEYWORD" >> "%LOG_FILE%" 2>&1
IF ERRORLEVEL 1 (
    SET "FAILURE_MESSAGE=Launcher path preparation failed."
    GOTO :fail
)

"%PYTHON_EXE%" "%PROJECT_ROOT%\hide_code_cells.py" "%PROJECT_ROOT%" >> "%LOG_FILE%" 2>&1
IF ERRORLEVEL 1 (
    SET "FAILURE_MESSAGE=Notebook metadata preparation failed."
    GOTO :fail
)

REM Keep conda verification enabled, but avoid the Windows trust store that
REM failed during installation. Conda accepts a CA bundle path for ssl_verify.
echo Configuring conda to use verified CA bundle: "%CA_BUNDLE%" >> "%LOG_FILE%"
"%PYTHON_EXE%" -m conda config --system --set ssl_verify "%CA_BUNDLE%" >> "%LOG_FILE%" 2>&1
IF ERRORLEVEL 1 (
    echo WARNING: Could not configure conda with the selected CA bundle. The application launcher still has its own verified fallback. >> "%LOG_FILE%"
)

REM Validate TLS and Jupyter imports before creating shortcuts or registering
REM the application.
"%PYTHON_EXE%" "%PROJECT_ROOT%\launch_jupyter.py" --self-test >> "%LOG_FILE%" 2>&1
IF ERRORLEVEL 1 (
    SET "FAILURE_MESSAGE=Launcher preflight failed. See menuinst_debug.log and launcher_debug.log."
    GOTO :fail
)
echo Launcher preflight completed successfully. >> "%LOG_FILE%"

"%PYTHON_EXE%" -c "import os, sys; print('Python:', sys.executable); print('Prefix:', os.environ.get('PREFIX'))" >> "%LOG_FILE%" 2>&1
IF ERRORLEVEL 1 (
    SET "FAILURE_MESSAGE=Python installation diagnostics failed."
    GOTO :fail
)

"%PYTHON_EXE%" -c "from menuinst.api import install; import os; print(install(os.path.join(r'%PREFIX%', 'PROJECT_NAME', 'notebook_launcher.json')))" >> "%LOG_FILE%" 2>&1
IF ERRORLEVEL 1 (
    SET "FAILURE_MESSAGE=Application shortcut creation failed."
    GOTO :fail
)

SET "ARP_KEY=HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\UNDERSCORED_PROJECT_NAME"
SET "UNINSTALL_EXE=%PREFIX%\Uninstall-UNDERSCORED_PROJECT_NAME.exe"
SET "DISPLAY_ICON=%PROJECT_ROOT%\logo.ico"
SET "DISPLAY_VERSION=VERSION_NUMBER"
SET "PUBLISHER=GITHUB_OWNER"
echo Registering PROJECT_NAME in Windows Apps list >> "%LOG_FILE%"
reg add "%ARP_KEY%" /v DisplayName /d "PROJECT_NAME" /f >> "%LOG_FILE%" 2>&1
reg add "%ARP_KEY%" /v DisplayVersion /d "%DISPLAY_VERSION%" /f >> "%LOG_FILE%" 2>&1
reg add "%ARP_KEY%" /v Publisher /d "%PUBLISHER%" /f >> "%LOG_FILE%" 2>&1
reg add "%ARP_KEY%" /v InstallLocation /d "%PREFIX%" /f >> "%LOG_FILE%" 2>&1
reg add "%ARP_KEY%" /v DisplayIcon /d "%DISPLAY_ICON%" /f >> "%LOG_FILE%" 2>&1
reg add "%ARP_KEY%" /v UninstallString /d "\"%UNINSTALL_EXE%\"" /f >> "%LOG_FILE%" 2>&1
reg add "%ARP_KEY%" /v QuietUninstallString /d "\"%UNINSTALL_EXE%\" /S" /f >> "%LOG_FILE%" 2>&1
reg add "%ARP_KEY%" /v NoModify /t REG_DWORD /d 1 /f >> "%LOG_FILE%" 2>&1
reg add "%ARP_KEY%" /v NoRepair /t REG_DWORD /d 1 /f >> "%LOG_FILE%" 2>&1

echo Post-install completed successfully. >> "%LOG_FILE%"
echo Post-install completed successfully.
ENDLOCAL
EXIT /B 0

:fail
IF EXIST "%CERTIFI_PATH_FILE%" DEL /Q "%CERTIFI_PATH_FILE%" >NUL 2>&1
IF NOT DEFINED FAILURE_MESSAGE SET "FAILURE_MESSAGE=Post-install failed for an unknown reason."
echo ERROR: %FAILURE_MESSAGE% >> "%LOG_FILE%"
echo ERROR: %FAILURE_MESSAGE%
echo See "%LOG_FILE%" for details.
ENDLOCAL
EXIT /B 1

:detect_nvidia_smi
FOR %%P IN (
    "%ProgramFiles%\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
    "%ProgramW6432%\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
    "%SystemRoot%\System32\nvidia-smi.exe"
    "%SystemRoot%\Sysnative\nvidia-smi.exe"
) DO (
    IF NOT DEFINED NVIDIA_SMI IF EXIST "%%~P" SET "NVIDIA_SMI=%%~P"
)
IF DEFINED NVIDIA_SMI GOTO :EOF

FOR /F "delims=" %%I IN ('where.exe nvidia-smi.exe 2^>NUL') DO (
    IF NOT DEFINED NVIDIA_SMI SET "NVIDIA_SMI=%%~fI"
)
GOTO :EOF

:install_lc_tools
REM Optional (LabConstrictor tools bridge): install labconstrictor-tools when the app's package ships a module named <package>_lc_tools.
REM Called before the import check of the package, which may import labconstrictor_tools at module level.
REM This step must never fail the installation: a failure is logged as a WARNING and LC_TOOLS_READY stays 0 (no registration later).
REM LC_TOOLS_SPEC can point to another source (wheel, git URL, mirror).
REM Default source: the GitHub archive of the labconstrictor-tools release TAG below (a plain zip: no git needed on the user's computer),
REM because the package is not on PyPI yet. Once it is, use "labconstrictor-tools" here. To move the pin, change only this tag.
SET "LC_TOOLS_PIN=v0.2.0-rc.1"
IF NOT DEFINED LC_TOOLS_SPEC SET "LC_TOOLS_SPEC=https://github.com/CellMigrationLab/LabConstrictor-Tools/archive/refs/tags/%LC_TOOLS_PIN%.zip"
SET "LC_TOOLS_READY=0"
"%PYTHON_EXE%" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('PYTHON_PROJ_NAME_lc_tools') else 1)" >> "%LOG_FILE%" 2>&1
IF ERRORLEVEL 1 GOTO :EOF
echo Found PYTHON_PROJ_NAME_lc_tools: installing labconstrictor-tools before the import check of PROJECT_NAME. >> "%LOG_FILE%"
REM LC_TOOLS_SPEC goes to pip as one argument through Python: cmd.exe never re-parses it (a quote or & in it cannot run anything).
"%PYTHON_EXE%" -c "import os, subprocess, sys; sys.exit(subprocess.call([sys.executable, '-m', 'pip', 'install', os.environ['LC_TOOLS_SPEC']]))" >> "%LOG_FILE%" 2>&1
IF ERRORLEVEL 1 (
    echo WARNING: labconstrictor-tools could not be installed - see the pip output above in this file; the tools of PROJECT_NAME will not be registered ^(the import check fails only if the package needs labconstrictor-tools at import time^). >> "%LOG_FILE%"
    GOTO :EOF
)
SET "LC_TOOLS_READY=1"
GOTO :EOF
