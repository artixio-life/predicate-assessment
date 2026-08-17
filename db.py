import os
import logging

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


def get_db_connection():
    """
    Connect to the shared Postgres database — the same instance the
    source-predicate crawler and the regulatory-explorer app point at.
    This app only ever touches drug.products / drug.product_chunks (owned
    here) plus reads from source.drug_predicate_raw_records and
    drug.regulatory_geography (owned by those other repos).
    """
    try:
        return psycopg2.connect(
            host=os.getenv('POSTGRES_HOST'),
            port=os.getenv('POSTGRES_PORT'),
            database=os.getenv('POSTGRES_DB'),
            user=os.getenv('POSTGRES_USER'),
            password=os.getenv('POSTGRES_PASSWORD'),
        )
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise


def dict_cursor(conn):
    return conn.cursor(cursor_factory=RealDictCursor)


def init_db():
    """
    Apply schema/schema.sql. NOT called at startup — main.py leaves schema
    changes to be applied by hand. This exists for when you'd rather apply it
    from Python (`python -c 'import db; db.init_db()'`).

    Safe to re-run: every statement in schema.sql is idempotent. Requires that
    the source-predicate and regulatory-explorer schemas already exist on this
    database (schema.sql checks and raises a clear error naming which one is
    missing, rather than failing on a raw FK error).
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            schema_path = os.path.join(os.path.dirname(__file__), 'schema', 'schema.sql')
            with open(schema_path, 'r') as f:
                logger.info(f"Applying {schema_path}...")
                cur.execute(f.read())
                conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Error initializing database: {e}")
        raise
    finally:
        conn.close()
