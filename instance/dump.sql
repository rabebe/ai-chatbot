DROP TABLE IF EXISTS summary;
DROP TABLE IF EXISTS users;

CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  username VARCHAR(80) NOT NULL UNIQUE,
  email VARCHAR(120) NOT NULL UNIQUE,
  password_hash VARCHAR(512) NOT NULL
);

-- Summary table
CREATE TABLE summary (
  id SERIAL PRIMARY KEY,
  input_text TEXT NOT NULL,
  output_text TEXT,
  created_at TIMESTAMP,
  critique_text TEXT,
  score INTEGER,
  user_id INTEGER REFERENCES users(id)
);
