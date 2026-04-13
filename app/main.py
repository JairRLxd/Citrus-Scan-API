from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="Citrus Scan API",
    description="API académica para clasificación de naranjas y limones",
    version="0.2.0",
)

app.include_router(router)
