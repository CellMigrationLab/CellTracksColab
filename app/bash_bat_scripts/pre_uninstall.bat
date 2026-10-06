@ECHO ON
echo Running pre-uninstall
"%PREFIX%\python.exe" -m labconstrictor_tools unregister --name "CellTracksColab" --prefix "%PREFIX%" >NUL 2>&1
IF ERRORLEVEL 1 IF EXIST "%USERPROFILE%\.labconstrictor\apps\CellTracksColab.json" echo WARNING: could not remove CellTracksColab from the LabConstrictor tools list; if Napari or Fiji still list it, delete the files %USERPROFILE%\.labconstrictor\apps\CellTracksColab.json and CellTracksColab.schema.json. 1>&2
"%PREFIX%\python.exe" -c "from menuinst.api import remove; import os; remove(os.path.join(r'%PREFIX%', 'CellTracksColab', 'notebook_launcher.json'))"
SET "ARP_KEY=HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\CellTracksColab"
reg delete "%ARP_KEY%" /f >NUL 2>&1
echo Pre-uninstall completed!
SetLocal EnableDelayedExpansion
