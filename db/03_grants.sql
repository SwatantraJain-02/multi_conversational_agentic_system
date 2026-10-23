REVOKE CONNECT ON DATABASE northwind FROM PUBLIC;
GRANT  CONNECT ON DATABASE northwind TO agent_readonly;
GRANT  USAGE   ON SCHEMA public TO agent_readonly;
GRANT  SELECT  ON ALL TABLES IN SCHEMA public TO agent_readonly;
ALTER ROLE agent_readonly SET default_transaction_read_only = on;
ALTER ROLE agent_readonly SET statement_timeout = '5s';