from typing import Literal

import fastapi
import pandas as pd
import uvicorn
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator

from .model import DelayModel

app = fastapi.FastAPI()

model = DelayModel()


def hyphenize(field: str):
    return field.replace("_", "-")


@app.exception_handler(RequestValidationError)
async def validation_error_exception_handler(
    request: fastapi.Request, exc: RequestValidationError
):
    return JSONResponse(
        status_code=400,
        content={"message": str(exc)},
    )


class Flight(BaseModel):
    Fecha_I: str | None = None
    Vlo_I: int | None = None
    Ori_I: str | None = None
    Des_I: str | None = None
    Emp_I: str | None = None
    Fecha_O: str | None = None
    Vlo_O: str | None = None
    Ori_O: str | None = None
    Des_O: str | None = None
    Emp_O: str | None = None
    DIA: int | None = None
    MES: int | None = None
    AÑO: int | None = None
    DIANOM: str | None = None
    TIPOVUELO: Literal["I", "N"] | None = None
    OPERA: (
        Literal[
            "American Airlines",
            "Air Canada",
            "Air France",
            "Aeromexico",
            "Aerolineas Argentinas",
            "Austral",
            "Avianca",
            "Alitalia",
            "British Airways",
            "Copa Air",
            "Delta Air",
            "Gol Trans",
            "Iberia",
            "K.L.M.",
            "Qantas Airways",
            "United Airlines",
            "Grupo LATAM",
            "Sky Airline",
            "Latin American Wings",
            "Plus Ultra Lineas Aereas",
            "JetSmart SPA",
            "Oceanair Linhas Aereas",
            "Lacsa",
        ]
        | None
    ) = None
    SIGLAORI: str | None = None
    SIGLADES: str | None = None

    class Config:
        alias_generator = hyphenize

    @validator("MES")
    def validate_mes(cls, v):
        if v not in range(1, 13):
            raise ValueError("MES must be in range 1-12")
        return v


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
def post_predict(data: dict[Literal["flights"], list[Flight]]) -> dict:
    flights = data["flights"]
    data = pd.DataFrame([flight.model_dump(by_alias=True) for flight in flights])
    features = model.preprocess(data)
    target = model.predict(features)
    return {"predict": list(target)}


if __name__ == "__main__":
    uvicorn.run(
        "challenge.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
