from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import json
import logging
from typing import Dict, List, Set
import uuid
from datetime import datetime, timedelta
import redis
from contextlib import asynccontextmanager

import pandas as pd

def utc_now():
    """Get current UTC datetime"""
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc)

from config import *
from models import *
from ml_service import initialize_predictor, get_predictor
from worker import celery_app, predict_single_exoplanet, predict_batch_exoplanets

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Redis connection for real-time data
redis_client = redis.Redis.from_url(REDIS_URL, password=REDIS_PASSWORD, decode_responses=True)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.client_subscriptions: Dict[str, Set[str]] = {}  # client_id -> task_ids
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections.add(websocket)
        self.client_subscriptions[client_id] = set()
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, websocket: WebSocket, client_id: str):
        self.active_connections.discard(websocket)
        self.client_subscriptions.pop(client_id, None)
        logger.info(f"Client {client_id} disconnected")
    
    async def send_to_client(self, websocket: WebSocket, data: dict):
        try:
            await websocket.send_text(json.dumps(data))
        except Exception as e:
            logger.error(f"Error sending data to client: {e}")
    
    async def broadcast_task_update(self, task_id: str, data: dict):
        """Broadcast task updates to subscribed clients"""
        for client_id, task_ids in self.client_subscriptions.items():
            if task_id in task_ids:
                for websocket in self.active_connections.copy():
                    await self.send_to_client(websocket, {
                        "type": "task_update",
                        "task_id": task_id,
                        "data": data
                    })
    
    def subscribe_to_task(self, client_id: str, task_id: str):
        """Subscribe client to task updates"""
        if client_id in self.client_subscriptions:
            self.client_subscriptions[client_id].add(task_id)

manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting Exoplanet Detection API")
    
    # Initialize ML model
    if not initialize_predictor():
        logger.warning("Failed to initialize ML model - will try to load on first request")
    
    # Test Redis connection
    try:
        redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
    
    # Start background tasks
    asyncio.create_task(telemetry_monitor())
    
    yield
    
    # Shutdown
    logger.info("Shutting down Exoplanet Detection API")

# Create FastAPI app
app = FastAPI(
    title="Exoplanet Detection API",
    description="Distributed ML API for exoplanet detection using Random Forest",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def telemetry_monitor():
    """Background task to monitor and broadcast telemetry"""
    while True:
        try:
            # Get active tasks from Celery
            active_tasks = celery_app.control.inspect().active()
            
            if active_tasks:
                for worker, tasks in active_tasks.items():
                    for task in tasks:
                        task_id = task['id']
                        
                        # Get task result for progress updates
                        result = celery_app.AsyncResult(task_id)
                        
                        telemetry_data = {
                            "task_id": task_id,
                            "status": result.state,
                            "worker_id": worker,
                            "timestamp": utc_now().isoformat()
                        }
                        
                        if result.info and isinstance(result.info, dict):
                            telemetry_data.update(result.info)
                        
                        # Broadcast to subscribed clients
                        await manager.broadcast_task_update(task_id, telemetry_data)
            
            await asyncio.sleep(TELEMETRY_UPDATE_INTERVAL)
            
        except Exception as e:
            logger.error(f"Error in telemetry monitor: {e}")
            await asyncio.sleep(5)

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "subscribe_task":
                task_id = message.get("task_id")
                if task_id:
                    manager.subscribe_to_task(client_id, task_id)
                    await manager.send_to_client(websocket, {
                        "type": "subscription_confirmed",
                        "task_id": task_id
                    })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id)

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Exoplanet Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "task_status": "/tasks/{task_id}",
            "websocket": "/ws/{client_id}"
        }
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    try:
        # Check Redis connection
        redis_connected = redis_client.ping()
        
        # Check ML model
        predictor = get_predictor()
        model_loaded = predictor.is_loaded
        
        # Get worker stats
        worker_stats = celery_app.control.inspect().stats()
        active_workers = len(worker_stats) if worker_stats else 0
        
        # System metrics
        system_metrics = SystemMetrics(
            total_workers=active_workers,
            active_workers=active_workers,
            total_tasks_processed=0,  # Could be tracked in Redis
            average_processing_time_ms=0.0,  # Could be calculated from recent tasks
            current_queue_size=0,  # Could be queried from Celery
            system_load=0.0,  # Could be actual system load
            uptime_hours=0.0  # Could be tracked
        )
        
        return HealthCheck(
            status="healthy" if redis_connected and model_loaded else "degraded",
            timestamp=utc_now(),
            version="1.0.0",
            database_connected=True,  # If using a database
            redis_connected=redis_connected,
            active_workers=active_workers,
            system_metrics=system_metrics
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.post("/predict", response_model=SingleTaskResponse)
async def predict_single(request: ExoplanetPredictionRequest):
    """Submit a single prediction task"""
    try:
        # Submit task to Celery
        task = predict_single_exoplanet.delay(request.dict())
        
        return SingleTaskResponse(
            task_id=task.id,
            status=TaskStatus.PENDING,
            estimated_completion_time=utc_now() + timedelta(seconds=30),
            message=f"Prediction task submitted successfully. Task ID: {task.id}"
        )
        
    except Exception as e:
        logger.error(f"Error submitting prediction task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchTaskResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Submit a batch prediction task"""
    try:
        if len(request.predictions) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"Batch size too large. Maximum allowed: {MAX_BATCH_SIZE}"
            )
        
        batch_id = str(uuid.uuid4())
        prediction_data = [pred.dict() for pred in request.predictions]
        
        # Submit batch task
        task = predict_batch_exoplanets.delay(prediction_data, batch_id)
        
        estimated_time = utc_now() + timedelta(
            seconds=len(request.predictions) * 0.1  # Estimate 0.1 seconds per prediction
        )
        
        return BatchTaskResponse(
            batch_id=batch_id,
            task_ids=[task.id],
            estimated_completion_time=estimated_time,
            total_tasks=len(request.predictions),
            message=f"Batch prediction submitted successfully. Batch ID: {batch_id}"
        )
        
    except Exception as e:
        logger.error(f"Error submitting batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks/{task_id}", response_model=TaskResultResponse)
async def get_task_status(task_id: str):
    """Get task status and result"""
    try:
        result = celery_app.AsyncResult(task_id)
        
        # Create telemetry
        telemetry = TaskTelemetry(
            task_id=task_id,
            status=TaskStatus(result.state.lower()),
            progress=100 if result.state == "SUCCESS" else 0,
            worker_id=None,  # Could be extracted from result info
            started_at=None,  # Could be tracked
            completed_at=utc_now() if result.state == "SUCCESS" else None
        )
        
        # Extract result data
        prediction_result = None
        if result.state == "SUCCESS" and result.result:
            if isinstance(result.result, dict) and 'result' in result.result:
                prediction_result = PredictionResult(**result.result['result'])
        
        return TaskResultResponse(
            task_id=task_id,
            status=TaskStatus(result.state.lower()),
            result=prediction_result,
            telemetry=telemetry,
            created_at=utc_now(),  # Could be actual creation time
            updated_at=utc_now()
        )
        
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workers/status")
async def get_worker_status():
    """Get status of all workers"""
    try:
        stats = celery_app.control.inspect().stats()
        active = celery_app.control.inspect().active()
        
        workers = []
        if stats:
            for worker_name, worker_stats in stats.items():
                worker_active_tasks = len(active.get(worker_name, [])) if active else 0
                
                workers.append({
                    "worker_id": worker_name,
                    "status": "online",
                    "current_tasks": worker_active_tasks,
                    "max_capacity": worker_stats.get('pool', {}).get('max-concurrency', 1),
                    "last_heartbeat": utc_now(),
                    "total_tasks": worker_stats.get('total', 0)
                })
        
        return {"workers": workers, "total_workers": len(workers)}
        
    except Exception as e:
        logger.error(f"Error getting worker status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded ML model"""
    try:
        predictor = get_predictor()
        return predictor.get_model_info()
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# New endpoint: CSV upload and batch prediction
@app.post("/predict/csv")
async def predict_from_csv(file: UploadFile = File(...)):
    """Accept a CSV file, run batch predictions, and return results"""
    try:
        # Read CSV into DataFrame
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))

        # Prepare requests for batch prediction
        predictor = get_predictor()
        if not predictor.is_loaded:
            predictor.load_model()

        # Map DataFrame rows to ExoplanetPredictionRequest
        requests = []
        # Get feature defaults (medians)
        feature_defaults = getattr(predictor, 'imputer', None)
        # If imputer is a SimpleImputer, get statistics_
        if feature_defaults and hasattr(feature_defaults, 'statistics_'):
            median_map = dict(zip(predictor.feature_columns, feature_defaults.statistics_))
        else:
            # Fallback to FEATURE_DEFAULTS from config
            from config import FEATURE_DEFAULTS
            median_map = FEATURE_DEFAULTS

        for _, row in df.iterrows():
            # Convert row to dict
            data = {col: row.get(col, None) for col in predictor.feature_columns if col != 'mission_encoded'}
            # Fill missing values with medians/defaults
            for col in predictor.feature_columns:
                if col == 'mission_encoded':
                    continue
                if data.get(col) is None or (isinstance(data.get(col), float) and pd.isna(data.get(col))):
                    data[col] = median_map.get(col, 0.0)
            # Assume 'mission' column exists or set default
            if 'mission' not in data:
                data['mission'] = 'KEPLER'
            req = ExoplanetPredictionRequest(**data)
            requests.append(req)

        # Run batch prediction
        results = predictor.predict_batch(requests)

        # Format output: CONFIRMED or FALSE POSITIVE + confidence
        output = []
        for i, res in enumerate(results):
            label = 'CONFIRMED' if res.prediction == 1 else 'FALSE POSITIVE'
            output.append({
                'row': i + 1,
                'label': label,
                'confidence_score': res.probability,
                'confidence_level': res.confidence_level
            })

        return {'results': output}
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level=LOG_LEVEL.lower()
    )
