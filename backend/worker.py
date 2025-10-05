from celery import Celery
import logging
from typing import List, Dict, Any
import time
import uuid
from datetime import datetime

from config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND
from models import ExoplanetPredictionRequest, PredictionResult, TaskTelemetry, TaskStatus
from ml_service import get_torch_manager, request_to_feature_list, initialize_torch_models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    'exoplanet_worker',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_max_tasks_per_child=1000,
    worker_prefetch_multiplier=1,
)

@celery_app.task(bind=True, name='predict_single_exoplanet')
def predict_single_exoplanet(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Celery task for single exoplanet prediction
    """
    task_id = self.request.id
    logger.info(f"Starting prediction task {task_id}")

    try:
        # initial state
        self.update_state(state=TaskStatus.PROCESSING.value, meta={
            'status': TaskStatus.PROCESSING.value,
            'progress': 10,
            'message': 'Initializing prediction...'
        })

        # Load models
        initialize_torch_models()
        manager = get_torch_manager()

        self.update_state(state=TaskStatus.PROCESSING.value, meta={
            'status': TaskStatus.PROCESSING.value,
            'progress': 30,
            'message': 'Model loaded, preparing data...'
        })

        # Create prediction request
        request = ExoplanetPredictionRequest(**prediction_data)

        self.update_state(state=TaskStatus.PROCESSING.value, meta={
            'status': TaskStatus.PROCESSING.value,
            'progress': 60,
            'message': 'Making prediction...'
        })

        # Predict
        features = request_to_feature_list(request, target_length=manager.tabular_input_size)
        prob = manager.predict_tabular_array(features)

        result = {
            'prediction': int(prob > 0.5),
            'probability': float(prob),
            'confidence_level': 'High' if prob >= 0.8 or prob <= 0.2 else 'Medium' if prob >= 0.6 or prob <= 0.4 else 'Low',
            'processing_time_ms': 0.0
        }

        self.update_state(state=TaskStatus.PROCESSING.value, meta={
            'status': TaskStatus.PROCESSING.value,
            'progress': 90,
            'message': 'Finalizing results...'
        })

        logger.info(f"Prediction completed for task {task_id}")

        return {
            'status': TaskStatus.COMPLETED.value,
            'result': result,
            'task_id': task_id,
            'completed_at': datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in prediction task {task_id}: {str(e)}")
        self.update_state(state=TaskStatus.FAILED.value, meta={
            'status': TaskStatus.FAILED.value,
            'error': str(e),
            'task_id': task_id
        })
        raise e

@celery_app.task(bind=True, name='predict_batch_exoplanets')
def predict_batch_exoplanets(self, batch_data: List[Dict[str, Any]], batch_id: str = None) -> Dict[str, Any]:
    """
    Celery task for batch exoplanet predictions
    """
    task_id = self.request.id
    batch_id = batch_id or str(uuid.uuid4())
    total_predictions = len(batch_data)
    
    logger.info(f"Starting batch prediction task {task_id} with {total_predictions} predictions")
    
    try:
        # Update task status
        self.update_state(state=TaskStatus.PROCESSING.value, meta={
            'status': TaskStatus.PROCESSING.value,
            'progress': 5,
            'message': f'Initializing batch prediction for {total_predictions} items...',
            'batch_id': batch_id,
            'total_items': total_predictions
        })

        # Ensure torch models loaded
        initialize_torch_models()
        manager = get_torch_manager()

        self.update_state(state=TaskStatus.PROCESSING.value, meta={
            'status': TaskStatus.PROCESSING.value,
            'progress': 15,
            'message': 'Model loaded, preparing batch data...',
            'batch_id': batch_id
        })

        # Create prediction requests
        requests = [ExoplanetPredictionRequest(**data) for data in batch_data]

        # Process in smaller chunks for better progress updates
        chunk_size = min(100, max(1, total_predictions // 10))
        results = []

        for i in range(0, len(requests), chunk_size):
            chunk = requests[i:i + chunk_size]
            # For each request in chunk, convert to features and predict
            for req in chunk:
                features = request_to_feature_list(req, target_length=manager.tabular_input_size)
                prob = manager.predict_tabular_array(features)
                results.append({
                    'prediction': int(prob > 0.5),
                    'probability': float(prob)
                })

            # Update progress
            progress = 15 + (70 * (i + len(chunk)) / total_predictions)
            self.update_state(state=TaskStatus.PROCESSING.value, meta={
                'status': TaskStatus.PROCESSING.value,
                'progress': int(progress),
                'message': f'Processed {i + len(chunk)}/{total_predictions} predictions...',
                'batch_id': batch_id,
                'completed_items': i + len(chunk)
            })

        # Final update
        self.update_state(state=TaskStatus.PROCESSING.value, meta={
            'status': TaskStatus.PROCESSING.value,
            'progress': 95,
            'message': 'Finalizing batch results...',
            'batch_id': batch_id
        })

        logger.info(f"Batch prediction completed for task {task_id}")

        return {
            'status': TaskStatus.COMPLETED.value,
            'results': results,
            'batch_id': batch_id,
            'task_id': task_id,
            'total_processed': len(results),
            'completed_at': datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in batch prediction task {task_id}: {str(e)}")
        self.update_state(state=TaskStatus.FAILED.value, meta={
            'status': TaskStatus.FAILED.value,
            'error': str(e),
            'task_id': task_id,
            'batch_id': batch_id
        })
        raise e

@celery_app.task(name='health_check')
def health_check() -> Dict[str, Any]:
    """
    Health check task for workers
    """
    try:
        initialize_torch_models()
        manager = get_torch_manager()
        model_status = manager.is_loaded
        
        return {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'model_loaded': model_status,
            'worker_id': celery_app.current_worker_task.request.hostname if hasattr(celery_app, 'current_worker_task') else 'unknown'
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }

# Task routing (for multiple worker types if needed)
celery_app.conf.task_routes = {
    'predict_single_exoplanet': {'queue': 'prediction'},
    'predict_batch_exoplanets': {'queue': 'batch_prediction'},
    'health_check': {'queue': 'monitoring'}
}

if __name__ == '__main__':
    # Start worker with: celery -A worker worker --loglevel=info
    celery_app.start()
