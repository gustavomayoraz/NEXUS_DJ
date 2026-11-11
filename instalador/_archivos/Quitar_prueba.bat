@echo off
rem Borra lo que instalo el instalador (la carpeta %LOCALAPPDATA%\NEXUS y los accesos directos). No toca tu carpeta de desarrollo.
echo Se va a borrar: %LOCALAPPDATA%\NEXUS y los accesos directos de NEXUS (Escritorio y menu Inicio).
choice /m "Continuar"
if errorlevel 2 exit /b 0
rmdir /s /q "%LOCALAPPDATA%\NEXUS"
del "%USERPROFILE%\Desktop\NEXUS.lnk" 2>nul
del "%APPDATA%\Microsoft\Windows\Start Menu\Programs\NEXUS.lnk" 2>nul
echo Listo.
pause
