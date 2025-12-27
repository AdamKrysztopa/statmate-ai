#!/usr/bin/env python3
"""Initialize the StatmateAI database.

This script creates all database tables and can optionally seed with sample data.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.session import drop_db, init_db


def main() -> None:
    """Initialize database with optional data seeding."""
    parser = argparse.ArgumentParser(description='Initialize StatmateAI database')
    parser.add_argument('--drop', action='store_true', help='Drop existing tables before creating')
    parser.add_argument('--seed', action='store_true', help='Seed database with sample data')

    args = parser.parse_args()

    if args.drop:
        print('⚠️  Dropping existing tables...')
        drop_db()
        print('✅ Tables dropped')

    print('🔨 Creating database tables...')
    init_db()
    print('✅ Database initialized successfully')

    if args.seed:
        print('🌱 Seeding database with sample data...')
        from scripts.seed_db import seed_database

        seed_database()
        print('✅ Database seeded successfully')

    print('\n🎉 Database setup complete!')
    print('You can now start the API server with: uvicorn statmate.api.main:app --reload')


if __name__ == '__main__':
    main()
