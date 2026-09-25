@echo off
REM Initialise this folder as a git repo and push it to GitHub.
REM Usage:  init_and_push.bat <github-username> [repo-name]
REM Create the EMPTY repo on github.com first (no README, no .gitignore, no license).

setlocal

if "%~1"=="" (
  echo Usage: init_and_push.bat ^<github-username^> [repo-name]
  echo Example: init_and_push.bat vivanshgarg05 hybrid-retrieval-qa
  exit /b 1
)

set USERNAME=%~1
set REPO=%~2
if "%REPO%"=="" set REPO=hybrid-retrieval-qa

where git >nul 2>&1
if errorlevel 1 (
  echo Git is not installed or not on PATH. Install Git for Windows first.
  exit /b 1
)

cd /d "%~dp0"

if not exist ".git" (
  echo Initialising repository...
  git init
) else (
  echo Repository already initialised.
)

REM Set your identity if it has never been set globally.
git config user.name >nul 2>&1 || git config user.name "%USERNAME%"
git config user.email >nul 2>&1 || (
  echo Set your commit email first:  git config --global user.email "you@example.com"
  exit /b 1
)

git add .
git commit -m "Initial commit: hybrid retrieval pipeline, evaluation harness, paper skeleton" || echo Nothing new to commit.
git branch -M main

git remote get-url origin >nul 2>&1
if errorlevel 1 (
  git remote add origin https://github.com/%USERNAME%/%REPO%.git
) else (
  git remote set-url origin https://github.com/%USERNAME%/%REPO%.git
)

echo.
echo Pushing to https://github.com/%USERNAME%/%REPO%.git
echo When prompted for a password, paste your Personal Access Token.
echo.
git push -u origin main

endlocal
