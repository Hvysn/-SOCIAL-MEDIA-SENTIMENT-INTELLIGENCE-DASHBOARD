CREATE DATABASE IF NOT EXISTS social_analytics;
USE social_analytics;

CREATE TABLE final_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    source VARCHAR(50),
    clean_text TEXT,
    polarity FLOAT,
    sentiment VARCHAR(20),
    ml_prediction VARCHAR(20)
);