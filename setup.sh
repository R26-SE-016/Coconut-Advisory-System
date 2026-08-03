#!/bin/bash
# Quick Start Setup Script for SaruPol

echo "🥥 SaruPol - Setup"
echo "===================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.9+"
    exit 1
fi
echo -e "${GREEN}✓ Python $(python3 --version | cut -d' ' -f2) found${NC}"

# Setup Backend
echo -e "${BLUE}Setting up Backend...${NC}"
cd backend

# Create venv
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate venv
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt
echo -e "${GREEN}✓ Backend dependencies installed${NC}"

# Setup .env
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${YELLOW}⚠ Created .env file. Please edit it with your GROQ_API_KEY${NC}"
fi

# Return to root
cd ..

echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Edit backend/.env with your GROQ_API_KEY"
echo "2. Run backend: cd backend && source venv/bin/activate && python -m app.main"
echo "3. Configure API endpoint in app Settings"
echo ""
echo "API Docs will be available at: http://localhost:8000/docs"
