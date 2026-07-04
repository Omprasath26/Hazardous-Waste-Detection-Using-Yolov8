from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.predictor import save_upload_file, run_inference

app = FastAPI(title="Hazardous Waste Detection")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Mount static folder
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# ============================================================
# HOME PAGE
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "result": None,
            "uploaded_image_url": None,
            "output_image_url": None,
            "error": None
        }
    )


# ============================================================
# HEALTH CHECK
# ============================================================
@app.get("/health")
async def health():
    return {"status": "ok", "message": "Hazardous Waste Detection API is running"}


# ============================================================
# PREDICT PAGE
# ============================================================
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        if not file.filename:
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "request": request,
                    "result": None,
                    "uploaded_image_url": None,
                    "output_image_url": None,
                    "error": "No file selected."
                }
            )

        # Save uploaded file
        saved_image_path = save_upload_file(file)

        # Run inference
        result = run_inference(saved_image_path)

        if result is None:
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "request": request,
                    "result": None,
                    "uploaded_image_url": None,
                    "output_image_url": None,
                    "error": "Prediction failed."
                }
            )

        uploaded_image_url = f"/static/uploads/{result['input_image_name']}"
        output_image_url = f"/static/outputs/{result['output_image_name']}"

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "result": result,
                "uploaded_image_url": uploaded_image_url,
                "output_image_url": output_image_url,
                "error": None
            }
        )

    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "result": None,
                "uploaded_image_url": None,
                "output_image_url": None,
                "error": f"Error: {str(e)}"
            }
        )