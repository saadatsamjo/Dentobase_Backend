# app/database/connection.py

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from config.appconfig import settings



# For async operations, use create_async_engine and make sure your DB_URL starts with postgresql+asyncpg://
DATABASE_URL = settings.DB_URL

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    future=True,
    # echo=True,
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Base class for models
Base = declarative_base()

print(f"✅ Successfully Connected to database: {settings.DB_NAME}")

# Dependency to get database session
async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
            # print("Session committed successfully")
        except Exception as e:
            print(f"Error committing session in get_db: {e}")
            raise
        finally:
            await session.close()

