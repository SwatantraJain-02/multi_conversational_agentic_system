CREATE ROLE agent_app LOGIN PASSWORD :'app_pw';
CREATE ROLE agent_readonly LOGIN PASSWORD :'ro_pw';
CREATE DATABASE northwind;
CREATE DATABASE agent_memory OWNER agent_app;