#!/bin/sh
set -euo pipefail

DB_USER=${DB_USER:-dentobasedbuser}
DB_NAME=${DB_NAME:-dentobasedb}
SUPERUSER=${SUPERUSER:-saadat}

echo "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "STARTING CLEANUP PROCESS FOR '$DB_NAME' DATABASE!!!..."
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"

# Terminating all connections to the database.
echo "\n========================================================"
echo "Terminating all connections to the database: $DB_NAME..."
echo "========================================================\n"
psql -U "$SUPERUSER" -d postgres -c "
SELECT pg_terminate_backend(pg_stat_activity.pid)
FROM pg_stat_activity
WHERE pg_stat_activity.datname = '$DB_NAME'
  AND pid <> pg_backend_pid();"

# Dropping the database if it exists.
echo "\n========================================================"
echo "Dropping database: $DB_NAME (if it exists)..."
echo "========================================================\n"
psql -U "$SUPERUSER" -d postgres -c "DROP DATABASE IF EXISTS $DB_NAME;"

# Creating a new database with the specified owner.
echo "\n========================================================"
echo "Creating database: $DB_NAME for user: $DB_USER..."
echo "========================================================\n"
psql -U "$SUPERUSER" -d postgres -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;"

# Setting up database privileges - each with individual feedback
echo "\n========================================================"
echo "Setting up database privileges for user: $DB_USER..."
echo "========================================================\n"

echo "\n\nGranting CONNECT privilege to $DB_USER..."
psql -U "$SUPERUSER" -d "$DB_NAME" -c "GRANT CONNECT ON DATABASE $DB_NAME TO $DB_USER;"
echo "\n-----------------------------------------------------------"

echo "\n\nSetting public schema ownership to $DB_USER..."
psql -U "$SUPERUSER" -d "$DB_NAME" -c "ALTER SCHEMA public OWNER TO $DB_USER;"
echo "\n-----------------------------------------------------------"

echo "\n\nGranting ALL privileges on public schema to $DB_USER..."
psql -U "$SUPERUSER" -d "$DB_NAME" -c "GRANT ALL ON SCHEMA public TO $DB_USER;"
echo "\n-----------------------------------------------------------"

echo "\n\nGranting USAGE on public schema to $DB_USER..."
psql -U "$SUPERUSER" -d "$DB_NAME" -c "GRANT USAGE ON SCHEMA public TO $DB_USER;"
echo "\n-----------------------------------------------------------"

echo "\n\nSetting default table privileges for $DB_USER..."
psql -U "$SUPERUSER" -d "$DB_NAME" -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO $DB_USER;"
echo "\n-----------------------------------------------------------"

echo "\n\nSetting default sequence privileges for $DB_USER..."
psql -U "$SUPERUSER" -d "$DB_NAME" -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO $DB_USER;"
echo "\n-----------------------------------------------------------"

echo "\n\nSetting default function privileges for $DB_USER..."
psql -U "$SUPERUSER" -d "$DB_NAME" -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO $DB_USER;"
echo "\n-----------------------------------------------------------"

echo "\n\nSetting default type privileges for $DB_USER..."
psql -U "$SUPERUSER" -d "$DB_NAME" -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TYPES TO $DB_USER;"
echo "\n-----------------------------------------------------------"

# Removing old Alembic migration files (except __init__.py)
echo "\n========================================================"
echo "Removing old Alembic migration files..."
echo "========================================================\n"
find alembic/versions -type f -name "*.py" ! -name "__init__.py" -delete || true

# Cleaning the cache from alembic
echo "\n========================================================"
echo "Cleaning the cache from alembic..."
echo "========================================================\n"
find alembic -type d -name "__pycache__" -exec rm -rf {} + || true

# Cleaning the cache of the entire project
echo "\n========================================================"
echo "Cleaning the cache of the entire project..."
echo "========================================================\n"
find . -type d -name "__pycache__" -exec rm -rf {} + || true

# Stamping the database with the base revision
echo "\n========================================================"
echo "Stamping database with base revision..."
echo "========================================================\n"
alembic stamp base

# Autogenerating a new cleaned-up migration
echo "\n========================================================"
echo "Autogenerating cleaned-up migration..."
echo "========================================================\n"
alembic revision --autogenerate -m "DB Cleaned Up"

# Upgrading the database to the latest version
echo "\n========================================================"
echo "Upgrading database to the latest version..."
echo "========================================================\n"
alembic upgrade head

# Done
echo "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "✅✅ DATABASE '$DB_NAME' SUCCESSFULLY RESET AND MIGRATED!✅✅"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
