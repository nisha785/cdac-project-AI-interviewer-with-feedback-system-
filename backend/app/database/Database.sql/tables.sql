CREATE DATABASE IF NOT EXISTS ai_interview_db;
use ai_interview_db;

CREATE USER 'aiuser'@'localhost'
IDENTIFIED BY 'ai123';

GRANT ALL PRIVILEGES
ON ai_interview_db.*
TO 'aiuser'@'localhost';

FLUSH PRIVILEGES;

CREATE TABLE sessions (
    session_id INT AUTO_INCREMENT PRIMARY KEY,
    candidate_name VARCHAR(100),
    start_time DATETIME,
    end_time DATETIME,
    status VARCHAR(20)
);
CREATE TABLE questions (
    question_id INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT,
    question_text TEXT,
    question_number INT,

    FOREIGN KEY (session_id)
    REFERENCES sessions(session_id)
);
CREATE TABLE answers (
    answer_id INT AUTO_INCREMENT PRIMARY KEY,
    question_id INT,
    transcript TEXT,

    FOREIGN KEY (question_id)
    REFERENCES questions(question_id)
);
CREATE TABLE scores (
    score_id INT AUTO_INCREMENT PRIMARY KEY,
    question_id INT,

    accuracy_score FLOAT,
    speech_score FLOAT,
    facial_score FLOAT,

    FOREIGN KEY (question_id)
    REFERENCES questions(question_id)
);


