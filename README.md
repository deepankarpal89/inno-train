# InnoTone Training System

## Database Migrations

This project uses Tortoise ORM with Aerich for database migrations.

### Prerequisites

```bash
pip install aerich
```

### Initial Setup

1. Initialize Aerich (only needed once):
   ```bash
   aerich init -t models.database.TORTOISE_ORM
   ```
   This creates a `migrations` directory and `aerich.ini` configuration file.

### Creating Migrations

After modifying your models:

1. Generate a new migration:

   ```bash
   aerich migrate --name "your_migration_name"
   ```

   This creates a new migration file in the `migrations` directory.

2. Apply migrations to update your database:
   ```bash
   aerich upgrade
   ```

### Common Commands

- List all migrations:

  ```bash
  aerich history
  ```

- Apply specific migration:

  ```bash
  aerich upgrade [migration_name]
  ```

- Rollback last migration:
  ```bash
  aerich downgrade
  ```

### Troubleshooting

- If you encounter issues, check the generated migration files in the `migrations` directory
- Make sure your database connection is properly configured in `models/database.py`
- For complex migrations, you may need to manually edit the generated migration file

## Running Training Jobs

To run a training job in the background:

```bash
nohup ./run_docker_job.sh > training.log 2>&1 &
```

## Environment Setup

Add the project to your Python path:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
```
