@echo off
rem Starts the web app on this PC and opens it in Chrome. Close this window to stop it.
cd /d "%~dp0"
start "" "http://localhost:8080"
npx -y http-server@14.1.1 . -p 8080 -c-1
