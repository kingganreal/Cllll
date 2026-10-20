@echo off
rem Starts the app on this PC: the Windows app (puts clips on the desktop, tray icon)
rem and the web app at http://localhost:8080 (makes clips). Close this window to stop the web app.
cd /d "%~dp0"
set "APP=%LOCALAPPDATA%\Programs\fancam-stage-desktop\FancamStage.exe"
if exist "%APP%" goto installed
if not exist "desktop\node_modules\electron" (
  echo Preparing the Windows app for the first time, please wait...
  pushd desktop
  call npm install --no-audit --no-fund
  popd
)
start "" /min cmd /c "cd /d "%~dp0desktop" && npm start"
goto web
:installed
start "" "%APP%" --hidden
:web
start "" "http://localhost:8080"
echo Web app: http://localhost:8080  (close this window to stop it)
call npx -y http-server@14.1.1 . -p 8080 -c-1
