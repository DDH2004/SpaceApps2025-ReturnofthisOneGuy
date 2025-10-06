# Exoplanet Detection Platform - SpaceApps 2025

[![NASA SpaceApps Challenge](https://img.shields.io/badge/NASA-SpaceApps%202025-blue.svg)](https://www.spaceappschallenge.org/)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange.svg)](https://pytorch.org)
[![Next.js](https://img.shields.io/badge/Next.js-15.5.4-black.svg)](https://nextjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com)
<img width="1487" height="817" alt="Screenshot 2025-10-05 at 10 52 21 PM" src="https://github.com/user-attachments/assets/42760cee-4d75-40b0-9685-ff703cbf899f" />

> **Challenge**: Develop an advanced machine learning platform to identify exoplanets from Kepler and TESS mission data using state-of-the-art multimodal AI architectures.

## Project Overview

Launched: https://exoident.us/
This project implements a **comprehensive exoplanet detection system** combining multiple machine learning approaches with a modern web interface. The platform processes stellar and transit parameters from NASA's Kepler and TESS missions to identify potential exoplanets with high accuracy.

### Key Achievements

- ** Advanced AI Models **: Enhanced Multimodal Fusion (PyTorch) + Random Forest ensemble
- ** High Accuracy **: Up to 93.6% accuracy on tabular features, 87.2% on multimodal fusion  
- ** Distributed Architecture **: Scalable FastAPI backend with Celery workers
- ** Modern UI **: Responsive Next.js frontend with real-time predictions
- ** Production Ready **: Full Docker containerization and CI/CD support

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │────│   FastAPI       │────│   AI Models     │
│   (Next.js)     │    │   Backend       │    │   (PyTorch)     │
│                 │    │                 │    │                 │
│  • Real-time UI │    │  • REST API     │    │  • Multimodal   │
│  • WebSocket    │    │  • WebSocket    │    │  • Tabular      │
│  • Responsive   │    │  • Rate Limit   │    │  • Random Forest│
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                        │                        │
        │              ┌─────────────────┐    ┌─────────────────┐
        │              │     Redis       │    │   Workers       │
        └──────────────│  (Queue/Cache)  │────│   (Celery)      │
                       │                 │    │                 │
                       │  • Task Queue   │    │  • Distributed  │
                       │  • Rate Cache   │    │  • Scalable     │
                       │  • Sessions     │    │  • Monitoring   │
                       └─────────────────┘    └─────────────────┘
```

##  Features

### Machine Learning Models

| Model | Type | Accuracy | Parameters | Use Case |
|-------|------|----------|------------|----------|
| **Enhanced Multimodal** | PyTorch Fusion | 87.2% | 1.2M | Complex multimodal analysis |
| **Tabular PyTorch** | Deep Learning | 93.6% | 52K | Fast, reliable predictions |
| **Random Forest** | Ensemble | 85.0% | Variable | Baseline comparison |

### Data Processing

- **Sources**: Kepler KOI, TESS TOI candidate data
- **Features**: 39 stellar/transit parameters per candidate
- **Pipeline**: Automated preprocessing, scaling, missing value imputation
- **Format**: Support for both individual and batch predictions

  <img width="913" height="749" alt="wewe" src="https://github.com/user-attachments/assets/2c198013-c6dc-4ce4-b424-34260316d1aa" />


### Web Platform

- **Frontend**: Modern React interface with real-time updates
- **Backend**: FastAPI with automatic OpenAPI documentation
- **Real-time**: WebSocket connections for live prediction monitoring
- **Scalable**: Distributed worker architecture with Redis

## Project Structure

```
SpaceApps2025-ReturnofthisOneGuy/
├── frontend/                    # Next.js React application
│   ├── src/components/            # Reusable UI components
│   ├── src/pages/                 # Application pages
│   └── public/                    # Static assets
├── backend/                     # FastAPI distributed backend
│   ├── main.py                    # FastAPI application entry
│   ├── ml_service.py              # PyTorch model integration
│   ├── worker.py                  # Celery distributed workers
│   └── models.py                  # Pydantic data models
├    ── exoplanet-detection-pipeline/ # Advanced PyTorch models
│   ├── models/                    # Trained model files (.pth)
│   ├── src/                       # Model architectures
│   └── train_multimodal_enhanced.py # Training scripts
-----AI_Model_Forest/             # Random Forest models
│   ├── trained_model/             # Scikit-learn models
│   └── ml_model/                  # Training/inference code

├── data/                        # Kepler/TESS datasets
│   ├── combined_kepler_tess_exoplanets.csv
│   └── data_eng.ipynb             # Data engineering notebook
├── docker-compose.yml           # Multi-service orchestration
└── README.md                    # This file
```

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/DDH2004/SpaceApps2025-ReturnofthisOneGuy.git
cd SpaceApps2025-ReturnofthisOneGuy

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f backend
```

**Services Available:**
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Worker Monitor**: http://localhost:5555
- **WebSocket**: ws://localhost:8000/ws

### Option 2: Local Development

#### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Redis (required)
redis-server

# Start Celery worker
celery -A worker.celery_app worker --loglevel=info

# Start FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## Testing the Platform

### 1. **Health Check**
```bash
curl http://localhost:8000/health
```

### 2. **Single Prediction**
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "mission": "Kepler",
    "orbital_period_days": 4.887,
    "transit_duration_hours": 2.4,
    "transit_depth_ppm": 516.0,
    "planet_radius_re": 1.23,
    "equilibrium_temp_k": 1422.0,
    "insolation_flux_earth": 684.0,
    "stellar_teff_k": 6117.0,
    "stellar_radius_re": 1.34,
    "apparent_mag": 14.2,
    "ra": 291.93423,
    "dec": 48.14165
  }'
```

### 3. **Expected Response**
```json
{
  "prediction": 1,
  "probability": 0.8734,
  "confidence_level": "High",
  "processing_time_ms": 45.2,
  "model_used": "enhanced_multimodal"
}
```

## Model Details

### Enhanced Multimodal Fusion Architecture

```
Input Processing:
┌─────────────────┬─────────────────┬─────────────────┐
│   Tabular Net   │  ResidualCNN1D  │   PixelCNN2D    │
│   (39 features) │  (5, 128)       │  (32, 24, 24)   │
│        ↓        │        ↓        │        ↓        │
│  [256→128→64]   │  [32→64→128→256]│  [32→64→128]    │
│        ↓        │        ↓        │        ↓        │
│    1 output     │   64 features   │   64 features   │
└─────────────────┴─────────────────┴─────────────────┘
                              ↓
                        Fusion Network
                         [129→128→64→32→16→1]
                              ↓
                         Sigmoid → Probability
```

**Key Features:**
-  **Residual Connections**: Improved gradient flow
-  **Attention Mechanisms**: Focus on important features  
-  **Regularization**: Batch normalization + dropout
-  **Adaptive Pooling**: Variable input size handling
-  **Xavier Initialization**: Stable weight initialization

### Training Data

- **Total Samples**: 10,609 processed candidates
- **Kepler KOI**: Historical exoplanet candidates
- **TESS TOI**: Recent transit objects of interest
- **Features**: Orbital period, transit depth, stellar properties, etc.
- **Labels**: Confirmed exoplanets vs false positives

##  Performance Metrics

### Model Comparison

| Metric | Random Forest | Tabular PyTorch | Enhanced Multimodal |
|--------|---------------|-----------------|-------------------|
| **Accuracy** | 85.0% | 93.6% | 87.2% |
| **Precision** | 0.83 | 0.94 | 0.89 |
| **Recall** | 0.87 | 0.93 | 0.85 |
| **F1-Score** | 0.85 | 0.94 | 0.87 |
| **Inference Speed** | 12ms | 8ms | 35ms |
| **Memory Usage** | 50MB | 25MB | 150MB |

### Scalability Metrics

- **Throughput**: ~1000 predictions/second (distributed)
- **Latency**: <100ms average response time
- **Concurrency**: Supports 50+ simultaneous users
- **Worker Scaling**: Horizontal scaling via Celery

##  Development

### Adding New Models

1. **Create Model Architecture** in `exoplanet-detection-pipeline/src/models.py`
2. **Train Model** using provided training scripts
3. **Save Model** as `.pth` file in `models/` directory
4. **Update Backend** in `backend/ml_service.py` to load new model
5. **Test Integration** using provided test scripts

### API Extensions

The FastAPI backend supports easy extension:

```python
# Add new endpoint in backend/main.py
@app.post("/api/v1/custom-predict")
async def custom_prediction(request: CustomRequest):
    # Your custom logic here
    return {"result": "custom_prediction"}
```

### Frontend Customization

The Next.js frontend uses modern React patterns:

```jsx
// Add new component in frontend/src/components/
export function CustomComponent() {
  const [prediction, setPrediction] = useState(null);
  
  // Your component logic
  return <div>Custom Exoplanet Analysis</div>;
}
```

## Configuration

### Environment Variables

```bash
# Backend Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=true
LOG_LEVEL=INFO

# Redis Configuration  
REDIS_URL=redis://localhost:6379/0

# Model Paths
PYTORCH_MODEL_PATH=./exoplanet-detection-pipeline/models/enhanced_multimodal_fusion_model.pth
TABULAR_MODEL_PATH=./exoplanet-detection-pipeline/models/tabular_model.pth

# Frontend Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

## Monitoring & Observability

### Health Checks

- **Backend**: http://localhost:8000/health
- **Model Status**: http://localhost:8000/api/v1/model/info
- **Worker Monitor**: http://localhost:5555 (Celery Flower)

### Logging

- **Application Logs**: Structured JSON logging
- **Model Performance**: Prediction accuracy tracking
- **System Metrics**: Resource usage monitoring

### WebSocket Telemetry

Real-time monitoring of:
- Prediction requests and responses
- Model performance metrics
- System health status
- Worker task progress

## Deployment

### Production Deployment

1. **Configure Environment**
```bash
export DEBUG=false
export LOG_LEVEL=WARNING
export REDIS_URL=redis://your-redis-server:6379/0
```

2. **Scale Workers**
```bash
docker-compose up --scale celery_worker=4
```

3. **Monitor Performance**
```bash
docker-compose logs -f backend celery_worker
```

### Cloud Deployment

The platform supports deployment on:
- **AWS**: ECS, EKS, or EC2
- **Google Cloud**: GKE or Compute Engine  
- **Azure**: Container Instances or AKS
- **Local**: Docker Swarm or Kubernetes

##  Contributing

### Development Workflow

1. **Fork** the repository
2. **Create** feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** changes: `git commit -m 'Add amazing feature'`
4. **Push** to branch: `git push origin feature/amazing-feature`
5. **Open** Pull Request

### Code Standards

- **Python**: Follow PEP 8, use type hints
- **JavaScript**: ESLint + Prettier configuration
- **Testing**: Unit tests for all models and APIs
- **Documentation**: Clear docstrings and comments

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Team

**Team Name**: ReturnofthisOneGuy  
**Challenge**: NASA SpaceApps Challenge 2025

## Acknowledgments

- **NASA** for providing Kepler and TESS datasets
- **SpaceApps Challenge** for the inspiring challenge
- **PyTorch Community** for the deep learning framework
- **FastAPI** for the excellent web framework
- **Open Source Community** for countless tools and libraries

## Additional Resources

- **🔗 NASA Exoplanet Archive**: https://exoplanetarchive.ipac.caltech.edu/
- **🔗 TESS Mission**: https://tess.mit.edu/
- **🔗 Kepler Mission**: https://www.nasa.gov/mission_pages/kepler/main/index.html
- **🔗 PyTorch Documentation**: https://pytorch.org/docs/
- **🔗 FastAPI Documentation**: https://fastapi.tiangolo.com/



