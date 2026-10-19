"""Module 2: FastAPI Application Entrypoint.

Initializes the FastAPI application, CORS middleware, lifespan events,
and routes for the Nephron-Aware Renal Latent State Engine.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import router
from src.engine.patient_inference import PatientInferenceService
from src.engine.parameter_estimation import load_state_space_parameters
from src.engine.preprocessing import load_cohort_scalers


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Preload datasets and fitted parameters on startup into application state."""
    scaler_config = load_cohort_scalers()
    params = load_state_space_parameters()

    service = PatientInferenceService(
        scaler_config=scaler_config,
        state_space_params=params,
    )
    # Preload cohort tables in memory for sub-25ms response latency
    service.preload()
    app.state.inference_service = service
    yield


app = FastAPI(
    title="Nephron-Aware Renal Latent State Engine",
    description=(
        "Continuous-Discrete Multivariate State-Space Engine for Renal Latent Signals. "
        "Research digital twin prototype based on MIMIC-IV 3.1 longitudinal clinical data."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for Next.js frontend (default dev server: localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
