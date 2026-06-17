from fastapi import FastAPI, File, BackgroundTasks
from uvicorn import run as app_run
from fastapi.responses import Response, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.constants import APP_HOST, APP_PORT
from src.pipeline.training import TrainingPipeline
from src.pipeline.prediction import PredictionPipeline
from src.logger import logging
import threading

app = FastAPI(title="Signature Recognition System API")
app.mount("/static", StaticFiles(directory="frontend"), name="static")

origins = ['*']  # Allow all origins for development

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

training_status = {"status": "idle", "message": "", "progress": 0}
training_lock = threading.Lock()


def run_training_pipeline():
    """Run the training pipeline in the background"""
    global training_status
    with training_lock:
        try:
            training_status["status"] = "running"
            training_status["message"] = "Model training in progress..."
            training_status["progress"] = 0
            logging.info("Starting background training task")
            
            train_pipeline = TrainingPipeline()
            train_pipeline.run_pipeline()
            
            training_status["status"] = "completed"
            training_status["message"] = "Training completed successfully!"
            training_status["progress"] = 100
            logging.info("Training pipeline completed successfully")
        except Exception as e:
            error_msg = str(e)
            training_status["status"] = "failed"
            training_status["message"] = f"Training failed: {error_msg}"
            logging.error(f"Training pipeline failed: {error_msg}")


@app.get("/")
async def root():
    return JSONResponse({
        "message": "Signature Recognition System API",
        "endpoints": {
            "train": "GET /train - Start model training",
            "docs": "GET /docs - Interactive API documentation",
            "redoc": "GET /redoc - ReDoc documentation"
        }
    })


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.get("/ui")
async def frontend():
    return FileResponse("frontend/index.html")


@app.get("/train")
async def training(background_tasks: BackgroundTasks):
    """Start model training in the background"""
    try:
        with training_lock:
            if training_status["status"] == "running":
                return JSONResponse(
                    {"status": "already_running", "message": "Model training is already in progress"},
                    status_code=400
                )
        
        background_tasks.add_task(run_training_pipeline)
        logging.info("Training task queued")
        return JSONResponse(
            {"status": "started", "message": "Model training started in the background"}
        )

    except Exception as e:
        logging.error(f"Error starting training: {str(e)}")
        return JSONResponse(
            {"status": "error", "message": f"Error occurred! {e}"},
            status_code=500
        )


@app.get("/train/status")
async def training_status_endpoint():
    """Get the current status of model training"""
    return JSONResponse({
        "status": training_status["status"],
        "message": training_status["message"],
        "progress": training_status["progress"]
    })


@app.post("/predict")
async def prediction(image_file: bytes = File(description="A file read as bytes")):
    try:
        prediction_pipeline = PredictionPipeline()
        final_output = prediction_pipeline.run_pipeline(image_file)
        return final_output
    except Exception as e:
        return JSONResponse(content=f"Error Occurred! {e}", status_code=500)

if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
