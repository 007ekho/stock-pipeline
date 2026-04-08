from pydantic import BaseModel, field_validator
from typing import List

VALID_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}


class OrderbookEvent(BaseModel):
    symbol: str
    timestamp: int
    top_bid_price: float
    top_bid_qty: float
    top_ask_price: float
    top_ask_qty: float
    bid_depth: List[List[float]]
    ask_depth: List[List[float]]
    total_bid_volume: float
    total_ask_volume: float
    bid_ask_ratio: float

    @field_validator("symbol")
    @classmethod
    def symbol_must_be_known(cls, v: str) -> str:
        if v not in VALID_SYMBOLS:
            raise ValueError(f"Unknown symbol: {v}. Expected one of {VALID_SYMBOLS}")
        return v

    @field_validator("top_bid_price", "top_ask_price")
    @classmethod
    def price_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"price must be positive, got {v}")
        return v

    @field_validator("top_ask_price")
    @classmethod
    def ask_must_be_above_bid(cls, v: float, info) -> float:
        bid = info.data.get("top_bid_price")
        if bid is not None and v <= bid:
            raise ValueError(f"ask ({v}) must be greater than bid ({bid})")
        return v

    @field_validator("total_bid_volume", "total_ask_volume")
    @classmethod
    def volume_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"volume must be positive, got {v}")
        return v

    @field_validator("bid_ask_ratio")
    @classmethod
    def ratio_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"bid_ask_ratio must be positive, got {v}")
        return v

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_ms(cls, v: int) -> int:
        if v < 1_577_836_800_000:
            raise ValueError(f"timestamp looks wrong (too old): {v}")
        return v
