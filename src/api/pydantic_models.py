from pydantic import BaseModel

class RiskRequest(BaseModel):
    Frequency: int
    Monetary: float
    Recency: int
    TransactionHour: int
    TransactionDay: int
    TransactionMonth: int
    TransactionYear: int

class RiskResponse(BaseModel):
    risk_probability: float
