from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "training_job" (
    "uuid" CHAR(36) NOT NULL  PRIMARY KEY,
    "created_at" VARCHAR(32),
    "completed_at" VARCHAR(32),
    "project_id" VARCHAR(255) NOT NULL,
    "training_run_id" VARCHAR(255) NOT NULL,
    "project_yaml_config" JSON,
    "project_config" JSON,
    "training_request" JSON,
    "machine_config" JSON,
    "status" VARCHAR(9) NOT NULL  DEFAULT 'pending' /* PENDING: pending\nRUNNING: running\nCOMPLETED: completed\nFAILED: failed\nCANCELLED: cancelled */,
    "time_taken" REAL,
    "metadata" JSON
) /* Training Job Table - Top-level entity for training jobs */;
CREATE INDEX IF NOT EXISTS "idx_training_jo_trainin_cf5d39" ON "training_job" ("training_run_id");
CREATE TABLE IF NOT EXISTS "training_iteration" (
    "uuid" CHAR(36) NOT NULL  PRIMARY KEY,
    "created_at" VARCHAR(32),
    "completed_at" VARCHAR(32),
    "time_taken" REAL,
    "iteration_number" INT NOT NULL,
    "step_type" VARCHAR(21) NOT NULL  DEFAULT 'iteration' /* TRAJECTORY_GENERATION: trajectory_generation\nTRAINING: training\nEVALUATION: evaluation\nITERATION: iteration\nGROUP_ITERATION: group_iteration */,
    "step_config" JSON,
    "metadata" JSON,
    "training_job_id" CHAR(36) NOT NULL REFERENCES "training_job" ("uuid") ON DELETE CASCADE
) /* Training Iteration Table - Individual iterations within a training job */;
CREATE INDEX IF NOT EXISTS "idx_training_it_trainin_977a34" ON "training_iteration" ("training_job_id", "iteration_number");
CREATE TABLE IF NOT EXISTS "epoch_train" (
    "uuid" CHAR(36) NOT NULL  PRIMARY KEY,
    "created_at" VARCHAR(32),
    "completed_at" VARCHAR(32),
    "time_taken" REAL,
    "metadata" JSON,
    "iteration_number" INT NOT NULL,
    "epoch_number" INT NOT NULL,
    "model_path" VARCHAR(512),
    "optimizer_path" VARCHAR(512),
    "metrics" JSON,
    "iteration_id" CHAR(36) NOT NULL REFERENCES "training_iteration" ("uuid") ON DELETE CASCADE
) /* Epoch Train Table - Individual epochs within an iteration */;
CREATE INDEX IF NOT EXISTS "idx_epoch_train_iterati_5ed1f3" ON "epoch_train" ("iteration_id", "epoch_number");
CREATE TABLE IF NOT EXISTS "eval" (
    "uuid" CHAR(36) NOT NULL  PRIMARY KEY,
    "created_at" VARCHAR(32),
    "completed_at" VARCHAR(32),
    "time_taken" REAL,
    "model_id" VARCHAR(255) NOT NULL,
    "dataset" VARCHAR(255) NOT NULL,
    "config" JSON,
    "metrics" JSON,
    "eval_data_path" VARCHAR(512),
    "metadata" JSON,
    "iteration_id" CHAR(36) NOT NULL REFERENCES "training_iteration" ("uuid") ON DELETE CASCADE
) /* Eval Table - Evaluation results for models */;
CREATE INDEX IF NOT EXISTS "idx_eval_model_i_1ec421" ON "eval" ("model_id");
CREATE INDEX IF NOT EXISTS "idx_eval_iterati_bef885" ON "eval" ("iteration_id", "model_id");
CREATE TABLE IF NOT EXISTS "aerich" (
    "id" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    "version" VARCHAR(255) NOT NULL,
    "app" VARCHAR(100) NOT NULL,
    "content" JSON NOT NULL
);"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        """
