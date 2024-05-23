from collections import namedtuple


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


database_repositoriy_example = """from api_hub.database import models
from api_hub.database.repositories.generic import GenericDatabaseRepository
from sqlalchemy.ext.asyncio import AsyncSession


class ExposureDatabaseRepository(GenericDatabaseRepository):
    def __init__(self, session: AsyncSession, *args, **kwargs) -> None:
        super().__init__(models.Exposure, session, *args, **kwargs)"""

route_group_example = """{'routes': [{'route': '',
   'method': 'GET',
   'function': 'async def get_exposures(\n    page: int = 1,\n    page_size: int = 10,\n    sort: str = None,\n    db: AsyncSession = Depends(get_db_session),\n):\n    repo = ExposureDatabaseRepository(db)\n    items = await repo.filter(\n        page=page, page_size=page_size, sort=desc(sort) if sort else None\n    )\n    return items'},
  {'route': '/{pk}',
   'method': 'GET',
   'function': 'async def get_exposure(\n    pk: uuid.UUID,\n    db: AsyncSession = Depends(get_db_session),\n):\n    repo = ExposureDatabaseRepository(db)\n\n    try:\n        exposure = await repo.get(pk)\n    except Exception as e:\n        logger.error(f"Error getting exposure with id {pk}: {e}")\n        raise HTTPException(\n            status_code=500, detail=f"Error getting exposure with id {pk}"\n        )\n\n    if not exposure:\n        logger.error(f"Exposure with id {pk} not found")\n        raise HTTPException(status_code=404, detail=f"Exposure with id {pk} not found")\n\n    response = ExposureSchema.model_validate(exposure.__dict__)\n\n    return response'},
  {'route': '',
   'method': 'POST',
   'function': 'async def create_exposure(\n    exposure: ExposureSchema, db: AsyncSession = Depends(get_db_session)\n):\n    repo = ExposureDatabaseRepository(db)\n    try:\n        new_exposure = await repo.create(exposure.model_dump())\n    except Exception as e:\n        logger.error(f"Error creating exposure: {e}")\n        raise HTTPException(status_code=500, detail="Error creating exposure")\n\n    # we have to do this because the db has fields that are not in the schema\n    validated = ExposureSchema.model_validate(new_exposure.__dict__)\n\n    return validated'},
  {'route': '/{pk}',
   'method': 'PUT',
   'function': 'async def update_exposure(\n    pk: uuid.UUID,\n    exposure: ExposureSchema,\n    db: AsyncSession = Depends(get_db_session),\n):\n    repo = ExposureDatabaseRepository(db)\n\n    try:\n        updated_exposure = await repo.update(pk, exposure.model_dump())\n    except Exception as e:\n        logger.error(f"Error updating exposure with id {pk}: {e}")\n        raise HTTPException(\n            status_code=500,\n            detail=f"Error updating exposure with id {pk}. Message: {e}",\n        )\n\n    validated = ExposureSchema.model_validate(updated_exposure.__dict__)\n\n    return validated'},
  {'route': '/{pk}',
   'method': 'DELETE',
   'function': 'async def delete_exposure(\n    pk: uuid.UUID,\n    db: AsyncSession = Depends(get_db_session),\n):\n    repo = ExposureDatabaseRepository(db)\n\n    try:\n        await repo.delete(pk)\n    except Exception as e:\n        logger.error(f"Error deleting exposure with id {pk}. Message: {e}")\n        raise HTTPException(\n            status_code=500, detail=f"Error deleting exposure with id {pk}. Message {e}"\n        )\n\n    return JSONResponse(status_code=204, content="Exposure deleted")'}]}"""

test_example = """import pytest
import uuid
from httpx import AsyncClient
from fastapi import status
from functools import wraps

from api_hub.api.models import InjurySchema
from api_hub.testing import TestCrud


@pytest.fixture
def url() -> str:
    return "/api/v1/injuries/"


@pytest.fixture(scope="class")
def example_data():
    return {
        "id": str(uuid.uuid4()),
        "tekmir_harm_id": "301ec09d-ca25-47f4-abc9-4ecf05319c0c",
        "contact_id": "6c03dad0-9bac-42d7-b426-4b5c3f71dc2c",
        "discovery_of_injury_date": None,
        "discovery_of_injury_date_precision": None,
        "injury_date": None,
        "injury_date_precision": None,
        "substantiation_score": 90,
        "substantiation_score_time": "2024-04-25T18:39:26.298397",
    }

@pytest.fixture(scope="class")
def update_data():
    return { 'substantiation_score' : 80 }


@pytest.fixture(scope="class")
def model():
    return InjurySchema


class TestInjuryAPI:
    @pytest.mark.asyncio
    async def test_read_injuries(
        self, client: AsyncClient, url: str, model: InjurySchema
    ):
        response = await client.get(url)
        data = response.json()

        assert response.status_code == status.HTTP_200_OK
        assert isinstance(data, list), "Expected a list of injuries"

        for item in data:
            model(**item)

    @pytest.mark.asyncio
    async def test_crud(self, client: AsyncClient, url: str, example_data: dict, update_data: dict):
        await TestCrud.create(client, url, example_data)
        await TestCrud.read(client, url, example_data)
        await TestCrud.update(client, url, example_data, update_data)
        await TestCrud.delete(client, url, example_data)
"""

BrokenCodeExamples = namedtuple("BrokenCodeExamples", ['example_1', 'example_2', 'example_3', 'example_4', 'example_5'])

example_1 = """
def greet(name):
    print("Hello, " + name
"""

example_2 = """
def add_numbers(a, b):
    return a + b

result = add_numbers(5, "10")
"""

example_3 = """
class Dog:
    def __init__(self, name):
        self.name = name

    def bark(self):
        print("Woof!")

my_dog = Dog("Buddy")
my_dog.bark()
my_dog.wag_tail()
"""

example_4 = """
numbers = [1, 2, 3, 4, 5]
print(numbers[5])
"""

example_5 = """
def factorial(n):
    if n == 1:
        return 1
    else:
        return n * factorial(n - 1)

print(factorial(0))
"""

broken_code_examples = BrokenCodeExamples(example_1, example_2, example_3, example_4, example_5)