from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
import datetime

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ExoplanetPredictionRequest(BaseModel):
    """Request model for exoplanet prediction"""
    mission: str = Field(..., description="Mission type: Kepler or TESS")
    orbital_period_days: float = Field(..., gt=0)
    transit_duration_hours: float = Field(..., gt=0)
    transit_depth_ppm: float = Field(..., gt=0)
    planet_radius_re: Optional[float] = Field(None, gt=0)
    equilibrium_temp_k: Optional[float] = Field(None, gt=0)
    insolation_flux_earth: Optional[float] = Field(None, gt=0)
    stellar_teff_k: Optional[float] = Field(None, gt=0)
    stellar_radius_re: Optional[float] = Field(None, gt=0)
    apparent_mag: Optional[float] = Field(None)
    ra: Optional[float] = Field(None, ge=0, le=360)
    dec: Optional[float] = Field(None, ge=-90, le=90)

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    predictions: List[ExoplanetPredictionRequest]
    batch_name: Optional[str] = Field(None, description="Optional batch identifier")

class PredictionResult(BaseModel):
    """Result model for individual predictions"""
    prediction: int = Field(..., description="0 for False Positive, 1 for Exoplanet")
    probability: float = Field(..., ge=0, le=1, description="Confidence probability")
    confidence_level: str = Field(..., description="High/Medium/Low confidence")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    feature_analysis: Optional[Dict[str, Any]] = Field(None, description="Feature importance and analysis data")

class TaskTelemetry(BaseModel):
    """Telemetry data for task monitoring"""
    task_id: str
    status: TaskStatus
    progress: float = Field(..., ge=0, le=100, description="Progress percentage")
    worker_id: Optional[str] = Field(None, description="ID of worker processing the task")
    started_at: Optional[datetime.datetime] = None
    completed_at: Optional[datetime.datetime] = None
    error_message: Optional[str] = None
    estimated_completion: Optional[datetime.datetime] = None

class WorkerStatus(BaseModel):
    """Status information for a worker"""
    worker_id: str
    status: str = Field(..., description="online/offline/busy")
    current_tasks: int = Field(..., ge=0)
    max_capacity: int = Field(..., gt=0)
    last_heartbeat: datetime.datetime
    server_location: Optional[str] = None
    performance_metrics: Dict[str, float] = Field(default_factory=dict)

class BatchTaskResponse(BaseModel):
    """Response for batch task submission"""
    batch_id: str
    task_ids: List[str]
    estimated_completion_time: datetime.datetime
    total_tasks: int
    message: str

class SingleTaskResponse(BaseModel):
    """Response for single task submission"""
    task_id: str
    status: TaskStatus
    estimated_completion_time: datetime.datetime
    message: str

class TaskResultResponse(BaseModel):
    """Complete task result with metadata"""
    task_id: str
    batch_id: Optional[str] = None
    status: TaskStatus
    result: Optional[PredictionResult] = None
    telemetry: TaskTelemetry
    created_at: datetime.datetime
    updated_at: datetime.datetime

class SystemMetrics(BaseModel):
    """Overall system performance metrics"""
    total_workers: int
    active_workers: int
    total_tasks_processed: int
    average_processing_time_ms: float
    current_queue_size: int
    system_load: float = Field(..., ge=0, le=100)
    uptime_hours: float

class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime.datetime
    version: str
    database_connected: bool
    redis_connected: bool
    active_workers: int
    system_metrics: SystemMetrics
