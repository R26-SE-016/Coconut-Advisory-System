# SaruPol - Coconut Advisory System

A complete mobile application for coconut farming advisory using **React Native** frontend and **FastAPI** backend with RAG (Retrieval-Augmented Generation).

## Project Structure

```
coconut_advisory_system/
├── backend/                  # FastAPI Python backend
│   ├── app/
│   │   ├── main.py          # FastAPI application
│   │   └── __init__.py
│   ├── requirements.txt      # Python dependencies
│   ├── .env.example          # Environment template
│   └── README.md             # Backend documentation
├── frontend/                 # React Native mobile app
│   ├── src/
│   │   ├── screens/          # App screens (Chat, Settings)
│   │   ├── services/         # API and storage services
│   │   └── constants/        # App constants
│   ├── App.tsx               # Main app component
│   ├── package.json          # Node dependencies
│   ├── tsconfig.json         # TypeScript config
│   └── README.md             # Frontend documentation
├── step1_build_index.py      # FAISS index builder
├── step2_rag_engine.py       # RAG engine (used by backend)
├── step3_app.py              # Original Flask app (deprecated)
├── chat.html                 # Original web UI (deprecated)
├── knowledge_base/           # PDF documents for indexing
├── faiss_index/              # FAISS vector index
├── docker-compose.yml        # Docker setup (optional)
└── README.md                 # This file
```

## Quick Start

### Prerequisites

- **Backend**: Python 3.9+, pip
- **Frontend**: Node.js 16+, npm or yarn
- **Mobile Development**: 
  - Android: Android SDK, Android Studio
  - iOS: Xcode, Cocoa Pods

### Step 1: Set Up Backend

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Edit .env and add your GROQ_API_KEY
# Edit .env with your settings
```

### Step 2: Start Backend Server

```bash
# From backend directory with venv activated
python -m app.main

# Server will run at http://localhost:8000
# Visit http://localhost:8000/docs for API documentation
```

### Step 3: Set Up Frontend

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install
# or
yarn install

# For Android development
npm run android

# For iOS development
npm run ios

# Or start the Metro bundler
npm start
```

### Step 4: Configure Mobile App

1. Open the app and go to **Settings**
2. Set Backend URL: `http://your-backend-ip:8000`
3. Click "Test Connection"
4. Start asking questions!

## Features

### Backend (FastAPI)

✅ REST API endpoints for mobile clients  
✅ RAG-based question answering  
✅ FAISS vector search for document retrieval  
✅ Auto-generated API documentation (Swagger UI)  
✅ CORS support for cross-origin requests  
✅ Error handling and logging  
✅ Health check endpoints  
✅ Production-ready architecture  

### Frontend (React Native)

✅ Native mobile app for iOS and Android  
✅ Real-time chat interface  
✅ Persistent chat history  
✅ Source attribution for answers  
✅ Settings and configuration screen  
✅ API endpoint configuration  
✅ Offline support (local storage)  
✅ TypeScript for type safety  
✅ Responsive mobile UI  

## Configuration

### Backend Environment (.env)

```env
# API Server
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False

# LLM API
GROQ_API_KEY=your_groq_api_key_here

# Paths
KNOWLEDGE_BASE_DIR=knowledge_base
FAISS_INDEX_DIR=faiss_index

# CORS
ALLOWED_ORIGINS=http://localhost:8081,http://localhost:3000
```

### Frontend

Configure in the Settings screen:

- **Backend URL**: `http://localhost:8000` (for emulator/simulator)
- **For physical devices**: `http://your-machine-ip:8000`

**Note for Android emulator:**
- Use `10.0.2.2` instead of `localhost`
- Example: `http://10.0.2.2:8000`

## API Endpoints

### Health & Info

```
GET  /          # Root endpoint
GET  /health    # Health status
GET  /info      # System information
```

### Advisory

```
POST /ask       # Submit question and get answer
```

**Request:**
```json
{
  "question": "How do I maintain coconut trees?",
  "context": "optional additional context"
}
```

**Response:**
```json
{
  "success": true,
  "question": "How do I maintain coconut trees?",
  "answer": "Coconut trees require...",
  "sources": [
    {
      "title": "coconut_care.pdf",
      "content": "Document preview...",
      "metadata": {"source": "coconut_care.pdf"}
    }
  ],
  "confidence": 0.85
}
```

## Development Workflow

### Backend Development

```bash
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run with auto-reload
DEBUG=True python -m app.main

# Visit API docs at http://localhost:8000/docs
```

### Frontend Development

```bash
cd frontend
npm start                # Start Metro bundler

# In another terminal:
npm run android          # Run on Android emulator
# or
npm run ios              # Run on iOS simulator
```

### Building for Production

**Backend:**
```bash
# Using Gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app.main:app
```

**Frontend - Android:**
```bash
npm run build-android
```

**Frontend - iOS:**
```bash
npm run build-ios
```

## Docker Deployment (Optional)

```bash
# Start both services with Docker Compose
docker-compose up

# Backend: http://localhost:8000
# API docs: http://localhost:8000/docs
```

See `docker-compose.yml` for configuration.

## Troubleshooting

### Connection Issues

**Problem:** App can't connect to backend

**Solutions:**
1. Check backend is running: `http://localhost:8000/health`
2. Verify API endpoint in app Settings
3. For emulator/simulator:
   - Android: Use `10.0.2.2` instead of `localhost`
   - iOS: Use `localhost` with `--port` forwarding
4. Check firewall rules

### Android Emulator

```bash
# Forward localhost to emulator
adb reverse tcp:8000 tcp:8000

# Then use: http://localhost:8000
```

### iOS Simulator

```bash
# iOS simulator can access localhost directly
# Use: http://localhost:8000
```

### RAG Chain Not Loading

1. Verify FAISS index exists: `faiss_index/index.faiss`
2. Check GROQ_API_KEY is set
3. Rebuild index if needed: `python step1_build_index.py`

## Project Architecture

```
Mobile App ──HTTP──┬── FastAPI Server ──── RAG Engine
                   │                          │
                   │                      FAISS Index
                   │                      Knowledge Base
                   └── Local Storage (Chat History)
```

## Performance Tips

### Backend

- Use multiple workers for production
- Enable caching for common questions
- Monitor RAG chain load time
- Consider GPU acceleration for inference

### Frontend

- Local storage for offline chat history
- Image lazy loading for faster startup
- Memory management for long chat sessions
- Battery-optimized polling

## Security Considerations

1. **API Security**
   - Set `ALLOWED_ORIGINS` for CORS
   - Use HTTPS in production
   - Implement rate limiting
   - Add authentication if needed

2. **Data Privacy**
   - Store sensitive keys in .env (not in repo)
   - Use secure storage for mobile app
   - Implement data encryption

3. **Production Deployment**
   - Use environment variables for secrets
   - Enable HTTPS/TLS
   - Set DEBUG=False
   - Use strong GROQ_API_KEY protection

## Next Steps

1. **Data Preparation**: Add your PDF knowledge base to `knowledge_base/`
2. **Index Building**: Run `python step1_build_index.py`
3. **Backend Testing**: Visit `http://localhost:8000/docs`
4. **Mobile Testing**: Test on emulator/simulator then physical device
5. **Deployment**: Deploy backend to cloud, build app for distribution

## Documentation

- [Backend Setup Guide](backend/README.md)
- [Frontend Setup Guide](frontend/README.md)
- [API Documentation](http://localhost:8000/docs) (when running)

## License

MIT

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review individual README files
3. Check logs in both backend and frontend
4. Verify environment configuration

---

**Built with ❤️ for coconut farmers in Sri Lanka**
