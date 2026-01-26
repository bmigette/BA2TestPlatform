# BA2ML - Deep Learning Financial Forecasting Platform

A comprehensive platform for training and evaluating deep learning models for financial forecasting using genetic optimization and strategy backtesting.

## Overview

This platform provides two main components:

1. **Model Builder** - Uses genetic optimization to build best-fit models for financial timeseries prediction
2. **Strategy Backtester** - Tests trained models against historical market data to evaluate performance

**Current Status: 229/231 features implemented (99.1%)**

## Features

### Dataset Preparation
- Multi-provider data fetching (Alpha Vantage, Yahoo Finance, Polygon.io, EODHD)
- Multi-timeframe technical indicators
- Fundamental and macro-economic data integration
- News sentiment analysis with ML models
- Interactive dataset visualization
- Export to CSV/Parquet formats

### Model Building
- State-of-the-art models: LSTM, N-BEATS, RNN, RCNN
- Genetic algorithm optimization for hyperparameters
- User-configurable prediction targets (profit %, max drawdown, timeframe)
- GPU-accelerated training with PyTorch
- Real-time training progress via WebSocket
- Optimization profiles for reusable configurations

### Strategy Backtesting
- Model-based trading strategies
- Configurable entry/exit conditions
- Advanced risk management (stop loss, take profit, trailing stop)
- Position sizing and leverage control
- Commission and slippage modeling
- Comprehensive performance metrics (Sharpe ratio, max drawdown, win rate)
- Interactive charts with trade markers

### User Interface
- Modern React + TypeScript frontend
- Tailwind CSS with shadcn/ui components
- Real-time updates and progress monitoring
- Dark/light theme support
- Responsive design for all screen sizes
- Interactive financial charts with Recharts

## Quick Start

### 1. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy and edit the environment file:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

Required API keys:
- `ALPHA_VANTAGE_API_KEY` - For stock data
- `FINNHUB_API_KEY` - For news data
- `FMP_API_KEY` - For fundamentals
- `FRED_API_KEY` - For macro data

### 3. Initialize Database

```bash
cd backend
source venv/bin/activate
python ../scripts/init_backend_db.py
```

### 4. Start Backend

```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

Backend available at: http://localhost:8002
API docs: http://localhost:8002/docs

### 5. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend available at: http://localhost:5173

## Project Structure

```
BA2MLTestPlatform/
├── backend/                 # FastAPI backend
│   ├── app/                 # Application code
│   │   ├── api/             # API endpoints
│   │   ├── models/          # SQLAlchemy models
│   │   ├── schemas/         # Pydantic schemas
│   │   └── services/        # Business logic
│   ├── dataproviders/       # Market data providers
│   ├── datasets/            # Dataset cache
│   ├── trained_models/      # Saved ML models
│   └── requirements.txt     # Python dependencies
├── frontend/                # React frontend
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── pages/           # Page components
│   │   └── services/        # API client
│   └── package.json         # Node dependencies
├── tests/                   # Test files
│   └── test_dataproviders.py
├── docs/                    # Documentation
│   ├── feature_list.json    # Feature tracking
│   ├── implementation/      # Session notes
│   └── spec/                # Specifications
├── scripts/                 # Utility scripts
├── logs/                    # Application logs
├── .env.example             # Environment template
├── QUICK_START.md           # Quick start guide
└── README.md                # This file
```

## Technology Stack

### Backend
- **FastAPI** - Web framework
- **PyTorch** - Deep learning
- **Darts** - Timeseries forecasting
- **DEAP/PyGAD** - Genetic algorithms
- **SQLAlchemy** - Database ORM
- **SQLite** - Database

### Frontend
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Recharts** - Charts

## Testing

Run backend tests:
```bash
cd backend
source venv/bin/activate
python ../tests/test_dataproviders.py
```

## Documentation

See the [docs/](docs/) folder for:
- Complete feature list and status
- Implementation session notes
- Original specifications

## Development

This project was built through autonomous multi-session development with Claude. See `docs/implementation/` for session handoffs and progress tracking.

## License

For educational and research purposes.
