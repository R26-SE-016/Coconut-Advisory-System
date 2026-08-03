@echo off
REM Quick Start Setup Script for CocoCastAI (Windows)

echo 🥥 CocoCastAI - Setup
echo ====================================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.9+
    exit /b 1
)
echo ✓ Python found

REM Setup Backend
echo.
echo Setting up Backend...
cd backend

if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
    echo ✓ Virtual environment created
)

REM Activate venv
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing Python dependencies...
pip install -r requirements.txt
echo ✓ Backend dependencies installed

REM Setup .env
if not exist ".env" (
    copy .env.example .env
    echo ⚠ Created .env file. Please edit it with your GROQ_API_KEY
)

cd ..

echo.
echo ✅ Setup complete!
echo.
echo Next steps:
echo 1. Edit backend\.env with your GROQ_API_KEY
echo 2. Run backend: cd backend ^&^& venv\Scripts\activate ^&^& python -m app.main
echo 3. Configure API endpoint in app Settings
echo.
echo API Docs will be available at: http://localhost:8000/docs
pause
