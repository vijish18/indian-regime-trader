-- Least-privilege database credentials (Phase 23).
--
-- Runs once, on an empty data directory, via the postgres image's
-- /docker-entrypoint-initdb.d hook. Re-running it against an existing
-- volume is a no-op because the hook only fires on initialisation --
-- which is why every statement below is written to be idempotent anyway,
-- so it can be applied by hand to an existing database.
--
-- Two roles, because they answer two different questions:
--
--   the superuser  "who is allowed to change the shape of the data?"
--   the app role   "who is allowed to change the data?"
--
-- The application only ever needs the second. Giving it the first means
-- that a SQL injection, a mistaken migration, or a compromised container
-- can DROP the order and fill history this system reconciles against
-- after a restart -- the exact record whose loss Phase 18 showed is
-- unrecoverable, because the broker's view and this system's view can
-- then never be compared again.

\set app_user `echo "${APP_DB_USER:-irt_app}"`
\set app_password `echo "${APP_DB_PASSWORD}"`

-- The application's role: it may connect and it may use the schema, and
-- that is all it is granted directly. Table privileges come from the
-- default-privileges grant below.
--
-- `\gexec` runs the text each SELECT produces, which is how the role
-- name and password get quoted by `format(%I, %L)` rather than pasted
-- into a string -- the same reason application code uses bound
-- parameters. The `WHERE NOT EXISTS` makes it re-runnable by hand
-- against a database that already has the role.
SELECT format(
    'CREATE ROLE %I LOGIN PASSWORD %L NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS',
    :'app_user', :'app_password'
)
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = :'app_user')
\gexec

-- Connect: yes. Create anything in this database: no.
SELECT format('REVOKE ALL ON DATABASE %I FROM PUBLIC', current_database()) \gexec
SELECT format('GRANT CONNECT ON DATABASE %I TO %I', current_database(), :'app_user') \gexec

-- PUBLIC's implicit CREATE on the public schema is removed: on PostgreSQL
-- 15+ this is already the default, and this statement makes it explicit
-- and true on an older server too.
REVOKE CREATE ON SCHEMA public FROM PUBLIC;

SELECT format('GRANT USAGE ON SCHEMA public TO %I', :'app_user') \gexec

-- Rows, not structure. No CREATE, no ALTER, no DROP, no TRUNCATE --
-- note in particular that TRUNCATE is withheld: it is the one
-- row-shaped privilege that can erase a whole table's history in a
-- single statement, and nothing this application does needs it.
SELECT format(
    'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO %I', :'app_user'
) \gexec
SELECT format('GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO %I', :'app_user') \gexec

-- The same privileges on tables the superuser creates later, so a
-- migration does not silently leave the application unable to read its
-- own new table.
SELECT format(
    'ALTER DEFAULT PRIVILEGES FOR ROLE %I IN SCHEMA public '
    'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO %I',
    current_user, :'app_user'
) \gexec
SELECT format(
    'ALTER DEFAULT PRIVILEGES FOR ROLE %I IN SCHEMA public '
    'GRANT USAGE, SELECT ON SEQUENCES TO %I',
    current_user, :'app_user'
) \gexec
