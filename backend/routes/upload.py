from fastapi import APIRouter, UploadFile, File, HTTPException
import tempfile
import os

from backend.services.csv_service import (
    validate_csv,
    get_dataset_summary,
)

router = APIRouter(
    prefix="/api/upload",
    tags=["Dataset Upload"],
)


@router.post("/csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    temp_path = None

    try:
        contents = await file.read()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".csv",
        ) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        df = validate_csv(temp_path)
        summary = get_dataset_summary(df)

        return {
            "success": True,
            "filename": file.filename,
            "summary": summary,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {exc}",
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)