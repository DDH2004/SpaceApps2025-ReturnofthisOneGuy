from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
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
    
    # Initialize ML models
    if not initialize_predictor():
        logger.warning("Failed to initialize main ML model - will try to load on first request")
    
    # Initialize TabularNet model
    try:
        from tabular_service import initialize_tabular_predictor
        if not initialize_tabular_predictor():
            logger.warning("Failed to initialize TabularNet model - will try to load on first request")
        else:
            logger.info("TabularNet model initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing TabularNet model: {e}")
    
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

# ============================================================================
# TABULAR MODEL SPECIFIC ENDPOINTS
# ============================================================================

@app.post("/tabular/predict", response_model=SingleTaskResponse)
async def predict_single_tabular(request: ExoplanetPredictionRequest):
    """Submit a single prediction task using TabularNet only"""
    try:
        from worker import predict_single_tabular
        
        # Submit task to Celery
        task = predict_single_tabular.delay(request.dict())
        
        return SingleTaskResponse(
            task_id=task.id,
            status=TaskStatus.PENDING,
            estimated_completion_time=utc_now() + timedelta(seconds=10),  # Faster for tabular-only
            message=f"TabularNet prediction task submitted successfully. Task ID: {task.id}"
        )
        
    except Exception as e:
        logger.error(f"Error submitting TabularNet prediction task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tabular/predict/batch", response_model=BatchTaskResponse)
async def predict_batch_tabular(request: BatchPredictionRequest):
    """Submit a batch prediction task using TabularNet only"""
    try:
        from worker import predict_batch_tabular
        
        if len(request.predictions) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"Batch size too large. Maximum allowed: {MAX_BATCH_SIZE}"
            )
        
        batch_id = str(uuid.uuid4())
        prediction_data = [pred.dict() for pred in request.predictions]
        
        # Submit batch task
        task = predict_batch_tabular.delay(prediction_data, batch_id)
        
        estimated_time = utc_now() + timedelta(
            seconds=len(request.predictions) * 0.05  # Faster estimate for tabular-only (0.05s per prediction)
        )
        
        return BatchTaskResponse(
            batch_id=batch_id,
            task_ids=[task.id],
            estimated_completion_time=estimated_time,
            total_tasks=len(request.predictions),
            message=f"TabularNet batch prediction submitted successfully. Batch ID: {batch_id}"
        )
        
    except Exception as e:
        logger.error(f"Error submitting TabularNet batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tabular/predict/sync", response_model=PredictionResult)
async def predict_single_tabular_sync(request: ExoplanetPredictionRequest):
    """Make a synchronous TabularNet prediction (for fast results)"""
    try:
        from tabular_service import get_tabular_predictor, initialize_tabular_predictor
        
        # Initialize tabular predictor if not already loaded
        tabular_predictor = get_tabular_predictor()
        if not tabular_predictor.is_loaded:
            logger.info("Initializing TabularNet predictor for sync prediction")
            if not initialize_tabular_predictor():
                raise HTTPException(status_code=503, detail="Failed to load TabularNet model")
        
        # Make direct prediction
        result = tabular_predictor.predict(request)
        
        logger.info(f"Sync TabularNet prediction completed in {result.processing_time_ms:.2f}ms")
        return result
        
    except Exception as e:
        logger.error(f"Error in sync TabularNet prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tabular/model/info")
async def get_tabular_model_info():
    """Get information about the TabularNet model"""
    try:
        from tabular_service import get_tabular_predictor, initialize_tabular_predictor
        
        # Initialize tabular predictor if not already loaded
        tabular_predictor = get_tabular_predictor()
        if not tabular_predictor.is_loaded:
            logger.info("Initializing TabularNet predictor for model info")
            if not initialize_tabular_predictor():
                return {"error": "Failed to load TabularNet model", "is_loaded": False}
        
        return tabular_predictor.get_model_info()
        
    except Exception as e:
        logger.error(f"Error getting TabularNet model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tabular/health")
async def tabular_health_check():
    """Health check specifically for TabularNet model"""
    try:
        from tabular_service import get_tabular_predictor
        
        tabular_predictor = get_tabular_predictor()
        
        health_data = {
            "status": "healthy" if tabular_predictor.is_loaded else "unhealthy",
            "model_loaded": tabular_predictor.is_loaded,
            "model_type": "TabularNet (PyTorch)",
            "device": str(tabular_predictor.device) if tabular_predictor.is_loaded else "unknown",
            "timestamp": utc_now().isoformat(),
            "accuracy": "93.6%",
            "parameters": "52,353" if tabular_predictor.is_loaded else "unknown"
        }
        
        if not tabular_predictor.is_loaded:
            logger.warning("TabularNet model health check failed - model not loaded")
            raise HTTPException(status_code=503, detail="TabularNet model not loaded")
            
        return health_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TabularNet health check failed: {e}")
        raise HTTPException(status_code=503, detail="TabularNet service unhealthy")

# ==================== ENHANCED MULTIMODAL FUSION ENDPOINTS ====================

@app.post("/multimodal/predict/sync", response_model=PredictionResult)
async def predict_multimodal_sync(
    request: ExoplanetPredictionRequest,
    cnn1d_data: Optional[List[List[float]]] = None,
    cnn2d_data: Optional[List[List[List[List[float]]]]] = None
):
    """
    Synchronous Enhanced Multimodal prediction (for low-latency requirements)
    
    Enhanced multimodal fusion combines:
    - Tabular features (39 features)
    - CNN1D light curve data (5 windows × 128 points)
    - CNN2D phase-folded images (32 phases × 24×24 pixels)
    
    Ideal for: Complex multimodal analysis with attention mechanisms
    Performance: ~200ms, 87.2% accuracy
    """
    try:
        from multimodal_service import get_multimodal_predictor
        predictor = get_multimodal_predictor()
        
        if not predictor.is_loaded:
            if not predictor.load_model():
                raise HTTPException(status_code=500, detail="Failed to load Enhanced Multimodal model")
        
        # Convert CNN data if provided
        import numpy as np
        cnn1d_array = np.array(cnn1d_data[0]) if cnn1d_data and len(cnn1d_data) > 0 else None
        cnn2d_array = np.array(cnn2d_data[0]) if cnn2d_data and len(cnn2d_data) > 0 else None
        
        result = predictor.predict(request, cnn1d_array, cnn2d_array)
        
        logger.info(f"Enhanced Multimodal sync prediction: {result.prediction} (p={result.probability:.3f}, {result.processing_time_ms:.1f}ms)")
        return result
        
    except Exception as e:
        logger.error(f"Enhanced Multimodal sync prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced Multimodal prediction failed: {str(e)}")

@app.post("/multimodal/predict", response_model=TaskResponse)
async def predict_multimodal_async(
    request: ExoplanetPredictionRequest,
    cnn1d_data: Optional[List[List[float]]] = None,
    cnn2d_data: Optional[List[List[List[List[float]]]]] = None
):
    """
    Asynchronous Enhanced Multimodal prediction using Celery
    
    Returns immediately with task_id for status checking
    Use /status/{task_id} to check progress and get results
    """
    try:
        from worker import predict_single_multimodal
        
        # Convert CNN data to serializable format
        cnn1d_list = cnn1d_data[0] if cnn1d_data and len(cnn1d_data) > 0 else None
        cnn2d_list = cnn2d_data[0] if cnn2d_data and len(cnn2d_data) > 0 else None
        
        task = predict_single_multimodal.delay(
            request.dict(),
            cnn1d_list,
            cnn2d_list
        )
        
        logger.info(f"Enhanced Multimodal async task created: {task.id}")
        
        return TaskResponse(
            task_id=task.id,
            status=TaskStatus.PENDING,
            message="Enhanced Multimodal prediction task queued",
            model_type="enhanced_multimodal_fusion"
        )
        
    except Exception as e:
        logger.error(f"Enhanced Multimodal async prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to queue Enhanced Multimodal prediction: {str(e)}")

@app.post("/multimodal/predict/batch", response_model=TaskResponse)
async def predict_multimodal_batch(
    requests: List[ExoplanetPredictionRequest],
    cnn1d_batch: Optional[List[List[List[float]]]] = None,
    cnn2d_batch: Optional[List[List[List[List[List[float]]]]]] = None,
    batch_id: Optional[str] = None
):
    """
    Enhanced Multimodal batch prediction using Celery
    
    Efficiently processes multiple predictions with multimodal data
    Returns task_id for monitoring batch progress
    """
    try:
        if len(requests) == 0:
            raise HTTPException(status_code=400, detail="No prediction requests provided")
        
        if len(requests) > 1000:
            raise HTTPException(status_code=400, detail="Batch size too large (max 1000)")
        
        from worker import predict_batch_multimodal
        
        # Convert to serializable format
        batch_data = [request.dict() for request in requests]
        batch_id = batch_id or str(uuid.uuid4())
        
        # Convert CNN data to serializable format
        cnn1d_serializable = cnn1d_batch if cnn1d_batch else None
        cnn2d_serializable = cnn2d_batch if cnn2d_batch else None
        
        task = predict_batch_multimodal.delay(
            batch_data,
            cnn1d_serializable,
            cnn2d_serializable,
            batch_id
        )
        
        logger.info(f"Enhanced Multimodal batch task created: {task.id} for {len(requests)} predictions")
        
        return TaskResponse(
            task_id=task.id,
            status=TaskStatus.PENDING,
            message=f"Enhanced Multimodal batch prediction queued for {len(requests)} items",
            batch_id=batch_id,
            total_items=len(requests),
            model_type="enhanced_multimodal_fusion"
        )
        
    except Exception as e:
        logger.error(f"Enhanced Multimodal batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to queue Enhanced Multimodal batch prediction: {str(e)}")

@app.get("/multimodal/model/info")
async def get_multimodal_model_info():
    """Get Enhanced Multimodal Fusion model information and specifications"""
    try:
        from multimodal_service import get_multimodal_predictor
        predictor = get_multimodal_predictor()
        
        return predictor.get_model_info()
        
    except Exception as e:
        logger.error(f"Error getting Enhanced Multimodal model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get Enhanced Multimodal model info: {str(e)}")

@app.post("/multimodal/analyze")
async def analyze_multimodal_components(
    request: ExoplanetPredictionRequest,
    cnn1d_data: Optional[List[List[float]]] = None,
    cnn2d_data: Optional[List[List[List[List[float]]]]] = None
):
    """
    Get individual component predictions from Enhanced Multimodal model
    
    Returns predictions from:
    - Tabular neural network
    - CNN1D light curve analysis  
    - CNN2D phase image analysis
    - Final fusion prediction
    """
    try:
        from multimodal_service import get_multimodal_predictor
        predictor = get_multimodal_predictor()
        
        if not predictor.is_loaded:
            if not predictor.load_model():
                raise HTTPException(status_code=500, detail="Failed to load Enhanced Multimodal model")
        
        # Convert CNN data if provided
        import numpy as np
        cnn1d_array = np.array(cnn1d_data[0]) if cnn1d_data and len(cnn1d_data) > 0 else None
        cnn2d_array = np.array(cnn2d_data[0]) if cnn2d_data and len(cnn2d_data) > 0 else None
        
        analysis = predictor.get_individual_predictions(request, cnn1d_array, cnn2d_array)
        
        logger.info(f"Enhanced Multimodal component analysis completed")
        return analysis
        
    except Exception as e:
        logger.error(f"Enhanced Multimodal analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced Multimodal analysis failed: {str(e)}")

@app.get("/multimodal/health")
async def multimodal_health_check():
    """Health check specifically for Enhanced Multimodal Fusion model"""
    try:
        from multimodal_service import get_multimodal_predictor
        
        multimodal_predictor = get_multimodal_predictor()
        
        health_data = {
            "status": "healthy" if multimodal_predictor.is_loaded else "unhealthy",
            "model_loaded": multimodal_predictor.is_loaded,
            "model_type": "Enhanced Multimodal Fusion (PyTorch)",
            "device": str(multimodal_predictor.device) if multimodal_predictor.is_loaded else "unknown",
            "timestamp": utc_now().isoformat(),
            "accuracy": "87.2%",
            "architecture": "Tabular + CNN1D + CNN2D Fusion",
            "parameters": multimodal_predictor.get_model_info().get("parameters", {}).get("total", "unknown") if multimodal_predictor.is_loaded else "unknown"
        }
        
        if not multimodal_predictor.is_loaded:
            logger.warning("Enhanced Multimodal model health check failed - model not loaded")
            raise HTTPException(status_code=503, detail="Enhanced Multimodal model not loaded")
            
        return health_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced Multimodal health check failed: {e}")
        raise HTTPException(status_code=503, detail="Enhanced Multimodal service unhealthy")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level=LOG_LEVEL.lower()
    )
