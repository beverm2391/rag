from uuid import UUID
from datetime import date
from typing import Optional
from pydantic import BaseModel
from enum import Enum
class DatePrecisionEnum(str, Enum):
    Year = "Year"
    Month = "Month"
    Day = "Day"
    Unknown = "Unknown"

class MedicalInjurySchema(BaseModel):
    id: UUID
    injury_id: UUID
    onset_date: Optional[date]
    onset_date_precision: Optional[DatePrecisionEnum]
    onset_state: Optional[str]
    is_diagnosed: Optional[bool] = False
    diagnosis_date: Optional[date]
    diagnosis_date_precision: Optional[DatePrecisionEnum]
    diagnosis_state: Optional[str]
    diagnosis_provider_id: Optional[UUID]
    injury_severity: Optional[str]
    medical_injury_sub_type: Optional[str]

    class Config:
        from_attributes = True
        use_enum_values = True