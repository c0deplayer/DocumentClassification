CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    file_name character varying(255) NOT NULL UNIQUE,
    file_path character varying(255) NOT NULL UNIQUE,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    classification character varying(255) NULL,
    summary TEXT NULL
);