from api_hub.database import models
from api_hub.database.repositories.generic import GenericDatabaseRepository
from sqlalchemy.ext.asyncio import AsyncSession

class MedicalInjuryDatabaseRepository(GenericDatabaseRepository):
    def __init__(self, session: AsyncSession, *args, **kwargs) -> None:
        super().__init__(models.MedicalInjury, session, *args, **kwargs)