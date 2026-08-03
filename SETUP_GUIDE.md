# Mobile App Setup Guide

This guide will help you set up and run the SaruPol mobile app locally.

## Prerequisites

Before starting, ensure you have:

- **Python 3.9+** - Download from https://www.python.org/
- **Node.js 16+** - Download from https://nodejs.org/
- **Android Studio** (for Android) or **Xcode** (for iOS)
- **Groq API Key** - Get free key from https://console.groq.com/

## Step 1: Quick Setup (Automated)

### Windows
```bash
setup.bat
```

### macOS/Linux
```bash
bash setup.sh
```

This will:
1. Create Python virtual environment
2. Install all Python dependencies
3. Install all Node dependencies
4. Create `.env` file from template

## Step 2: Configure Environment

Edit `backend/.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False
```

## Step 3: Start Backend

### Windows
```bash
cd backend
venv\Scripts\activate
python -m app.main
```

### macOS/Linux
```bash
cd backend
source venv/bin/activate
python -m app.main
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Verify Backend is Running

Visit: http://localhost:8000/health

You should see:
```json
{
  "status": "healthy",
  "rag_chain_loaded": true,
  "retriever_loaded": true
}
```

## Step 4: Start Frontend

In a new terminal:

```bash
cd frontend
npm start
```

This starts the Metro bundler.

## Step 5: Run Mobile App

### Android

In another terminal:

```bash
cd frontend
npm run android
```

This will:
1. Build the app
2. Deploy to emulator/device
3. Start the app automatically

### iOS

In another terminal:

```bash
cd frontend
npm run ios
```

## Step 6: Configure App

1. Open the app
2. Go to **Settings**
3. Set Backend URL:
   - **For emulator/simulator**: Leave as default
   - **For physical device**: Enter your computer's IP address
     - Windows: `ipconfig` (find IPv4 Address)
     - Mac/Linux: `ifconfig` or `hostname -I`
     - Example: `http://192.168.1.100:8000`

4. Click **Test Connection**
5. Start asking questions!

## Network Configuration

### Android Emulator

The emulator cannot reach `localhost`. Options:

**Option 1: Forward port (recommended)**
```bash
adb reverse tcp:8000 tcp:8000
```
Then use: `http://localhost:8000`

**Option 2: Use emulator gateway**
Use: `http://10.0.2.2:8000`

### iOS Simulator

The simulator can reach localhost directly:
Use: `http://localhost:8000`

### Physical Device

Find your computer's IP address:

**Windows:**
```bash
ipconfig
```
Look for "IPv4 Address"

**macOS/Linux:**
```bash
ifconfig
```
or
```bash
hostname -I
```

Then use: `http://[your-ip]:8000`

## Troubleshooting

### "Cannot connect to API"

1. Check backend is running:
   ```bash
   curl http://localhost:8000/health
   ```

2. For emulator, forward port:
   ```bash
   adb reverse tcp:8000 tcp:8000
   ```

3. Check firewall allows connections

### "RAG chain not loaded"

1. Check GROQ_API_KEY is set in `.env`
2. Verify FAISS index exists: `faiss_index/index.faiss`
3. Rebuild index if needed:
   ```bash
   python step1_build_index.py
   ```

### "Command not found: npm"

Install Node.js from https://nodejs.org/

### Metro bundler crashes

```bash
# Clear cache and reinstall
cd frontend
rm -rf node_modules
npm install
npm start
```

## Development Tips

### Hot Reload

Changes to frontend code will automatically reload in the app.

### Debug Mode

To see detailed logs:

**Backend:**
```bash
DEBUG=True python -m app.main
```

**Frontend:**
Open React Native debugger:
- Press `d` in terminal where Metro is running
- Or press `Cmd+D` (iOS) or `Cmd+M` (Android)

### Testing Locally

Visit API documentation:
http://localhost:8000/docs

Try the `/ask` endpoint directly with test questions.

## Next Steps

1. ✅ Customize the UI with your branding
2. ✅ Add more features (voice input, export, etc.)
3. ✅ Deploy backend to cloud
4. ✅ Build and publish to app stores

## Support

See:
- [Main README](README.md)
- [Backend README](backend/README.md)
- [Frontend README](frontend/README.md)
