from dataclasses import dataclass


database_model_example = """class Injury(Base):
    __tablename__ = "injury"
    __table_args__ = (
        ForeignKeyConstraint(
            ["contact_id"], ["contact.id"], name="injury_contact_id_fkey"
        ),
        ForeignKeyConstraint(
            ["tekmir_harm_id"], ["tekmir_harm.id"], name="injury_tekmir_harm_id_fkey"
        ),
        PrimaryKeyConstraint("id", name="medical_pkey"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, server_default=text("gen_random_uuid()")
    )
    tekmir_harm_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    contact_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    discovery_of_injury_date: Mapped[Optional[datetime.date]] = mapped_column(Date)
    discovery_of_injury_date_precision: Mapped[Optional[str]] = mapped_column(
        Enum("Year", "Month", "Day", "Unknown", name="date_precision_enum")
    )
    injury_date: Mapped[Optional[datetime.date]] = mapped_column(Date)
    injury_date_precision: Mapped[Optional[str]] = mapped_column(
        Enum("Year", "Month", "Day", "Unknown", name="date_precision_enum")
    )
    substantiation_score: Mapped[Optional[decimal.Decimal]] = mapped_column(Numeric)
    substantiation_score_time: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime
    )

    contact: Mapped["Contact"] = relationship("Contact", back_populates="injury")
    tekmir_harm: Mapped["TekmirHarm"] = relationship(
        "TekmirHarm", back_populates="injury"
    )
    injury_substantiation: Mapped[List["InjurySubstantiation"]] = relationship(
        "InjurySubstantiation", back_populates="injury"
    )
    medical_injury: Mapped[List["MedicalInjury"]] = relationship(
        "MedicalInjury", back_populates="injury"
    )
"""

pydantic_model_example = """
from uuid import UUID
from datetime import date, datetime
from typing import Optional
from pydantic import BaseModel
from enum import Enum

class DatePrecisionEnum(str, Enum):
    Year = "Year"
    Month = "Month"
    Day = "Day"
    Unknown = "Unknown"

class InjurySchema(BaseModel):
    id: UUID
    tekmir_harm_id: UUID
    contact_id: UUID
    discovery_of_injury_date: Optional[date]
    discovery_of_injury_date_precision: Optional[DatePrecisionEnum]
    injury_date: Optional[date]
    injury_date_precision: Optional[DatePrecisionEnum]
    substantiation_score: Optional[float] = None
    substantiation_score_time: Optional[datetime] = None

    class Config(ConfigDict):
        from_attributes = True # this is the updated way to use the from_orm method
        use_enum_values = True # this is for the Enum values to be used instead of the index
"""

broken_code = """from pydantic import BaseModel

class User(BaseModel):
    id: UUID
    name: str
    email: str
    is_active: Optional[bool] = True

    def greet(self)
        return f"Hello, {self.name}!"
"""