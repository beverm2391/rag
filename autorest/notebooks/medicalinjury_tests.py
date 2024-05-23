import pytest
import uuid
from httpx import AsyncClient
from fastapi import status
from functools import wraps

from api_hub.api.models import MedicalInjurySchema
from api_hub.testing import TestCrud

@pytest.fixture
def url() -> str:
    return "/apu/v1/medical-injury"


@pytest.fixture(scope="class")
def example_data():
    return {
        "id": str(uuid.uuid4()),
        "injury_id": "6416c516-8980-4105-95b6-6f38599ed3c8",
        "onset_date": None,
        "onset_date_precision": None,
        "onset_state": None,
        "is_diagnosed": True,
        "diagnosis_date": "2015-03-01",
        "diagnosis_date_precision": "Day",
        "diagnosis_state": None,
        "diagnosis_provider_id": None,
        "injury_severity": None,
        "medical_injury_sub_type": None,
    }

@pytest.fixture(scope="class")
def update_data():
    return { 'diagnosis_date': '2015-03-02' }


@pytest.fixture(scope="class")
def model():
    return MedicalInjurySchema


class TestMedicalInjuryAPI:
    @pytest.mark.asyncio
    async def test_read_medical_injuries(
        self, client: AsyncClient, url: str, model: MedicalInjurySchema
    ):
        response = await client.get(url)
        data = response.json()

        assert response.status_code == status.HTTP_200_OK
        assert isinstance(data, list), "Expected a list of medical injuries"

        for item in data:
            model(**item)

    @pytest.mark.asyncio
    async def test_crud(self, client: AsyncClient, url: str, example_data: dict, update_data: dict):
        await TestCrud.create(client, url, example_data)
        await TestCrud.read(client, url, example_data)
        await TestCrud.update(client, url, example_data, update_data)
        await TestCrud.delete(client, url, example_data)