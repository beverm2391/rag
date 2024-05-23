@router.GET('')
async def get_medical_injuries(
    page: int = 1,
    page_size: int = 10,
    sort: str = None,
    db: AsyncSession = Depends(get_db_session),
):
    repo = MedicalInjuryDatabaseRepository(db)
    items = await repo.filter(
        page=page, page_size=page_size, sort=desc(sort) if sort else None
    )
    return items

@router.GET('/{pk}')
async def get_medical_injury(
    pk: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
):
    repo = MedicalInjuryDatabaseRepository(db)

    try:
        medical_injury = await repo.get(pk)
    except Exception as e:
        logger.error(f"Error getting medical injury with id {pk}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting medical injury with id {pk}"
        )

    if not medical_injury:
        logger.error(f"Medical injury with id {pk} not found")
        raise HTTPException(status_code=404, detail=f"Medical injury with id {pk} not found")

    response = MedicalInjurySchema.model_validate(medical_injury.__dict__)

    return response

@router.POST('')
async def create_medical_injury(
    medical_injury: MedicalInjurySchema, db: AsyncSession = Depends(get_db_session)
):
    repo = MedicalInjuryDatabaseRepository(db)
    try:
        new_medical_injury = await repo.create(medical_injury.model_dump())
    except Exception as e:
        logger.error(f"Error creating medical injury: {e}")
        raise HTTPException(status_code=500, detail="Error creating medical injury")

    validated = MedicalInjurySchema.model_validate(new_medical_injury.__dict__)

    return validated

@router.PUT('/{pk}')
async def update_medical_injury(
    pk: uuid.UUID,
    medical_injury: MedicalInjurySchema,
    db: AsyncSession = Depends(get_db_session),
):
    repo = MedicalInjuryDatabaseRepository(db)

    try:
        updated_medical_injury = await repo.update(pk, medical_injury.model_dump())
    except Exception as e:
        logger.error(f"Error updating medical injury with id {pk}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating medical injury with id {pk}. Message: {e}",
        )

    validated = MedicalInjurySchema.model_validate(updated_medical_injury.__dict__)

    return validated

@router.DELETE('/{pk}')
async def delete_medical_injury(
    pk: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
):
    repo = MedicalInjuryDatabaseRepository(db)

    try:
        await repo.delete(pk)
    except Exception as e:
        logger.error(f"Error deleting medical injury with id {pk}. Message: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error deleting medical injury with id {pk}. Message {e}"
        )

    return JSONResponse(status_code=204, content="Medical injury deleted")

