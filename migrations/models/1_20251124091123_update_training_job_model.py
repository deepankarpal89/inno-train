from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "training_job" ADD "started_at" VARCHAR(32);
        ALTER TABLE "training_job" DROP COLUMN "project_config";"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "training_job" ADD "project_config" JSON;
        ALTER TABLE "training_job" DROP COLUMN "started_at";"""
