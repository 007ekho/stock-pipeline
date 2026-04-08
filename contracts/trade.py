from pydantic import BaseModel, field_validator
from typing import Literal

VALID_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}


class TradeEvent(BaseModel):
    symbol: str
    price: float
    size: float
    timestamp: int
    trade_id: int

    @field_validator("symbol")
    @classmethod
    def symbol_must_be_known(cls, v: str) -> str:
        if v not in VALID_SYMBOLS:
            raise ValueError(f"Unknown symbol: {v}. Expected one of {VALID_SYMBOLS}")
        return v

    @field_validator("price")
    @classmethod
    def price_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"price must be positive, got {v}")
        return v

    @field_validator("size")
    @classmethod
    def size_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"size must be positive, got {v}")
        return v

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_ms(cls, v: int) -> int:
        # Binance timestamps are in milliseconds — must be > year 2020 in ms
        if v < 1_577_836_800_000:
            raise ValueError(f"timestamp looks wrong (too old): {v}")
        return v
