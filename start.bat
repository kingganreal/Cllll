@echo off
chcp 65001 >nul
rem Starts 직캠 스테이지 on this PC: the Windows app (puts clips on the desktop, tray icon)
rem and the web app at http://localhost:8080 (makes clips). Close this window to stop the web app.
cd /d "%~dp0"
set "APP=%LOCALAPPDATA%\Programs\fancam-stage-desktop\FancamStage.exe"
if exist "%APP%" (
  rem The installed app (from GitHub Releases). Already running? It just stays as it is.
  start "" "%APP%" --hidden
) else (
  if not exist "desktop\node_modules\electron" (
    echo 처음 한 번 윈도우 앱을 준비하고 있어요. 잠시만 기다려 주세요...
    pushd desktop
    call npm install --no-audit --no-fund
    popd
  )
  start "" /min cmd /c "cd /d ""%~dp0desktop"" && npm start"
)
start "" "http://localhost:8080"
npx -y http-server@14.1.1 . -p 8080 -c-1
