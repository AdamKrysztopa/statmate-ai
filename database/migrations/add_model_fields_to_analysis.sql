-- Migration: Add model_name and provider fields to analyses table
-- Date: 2025-01-16
-- Description: Add columns to track which AI model was used for each analysis

-- Add model_name column
ALTER TABLE analyses ADD COLUMN model_name VARCHAR(100);

-- Add provider column
ALTER TABLE analyses ADD COLUMN provider VARCHAR(50);

-- Add comment
COMMENT ON COLUMN analyses.model_name IS 'AI model used for analysis (e.g., gpt-4o, deepseek-r1:8b)';
COMMENT ON COLUMN analyses.provider IS 'Model provider (e.g., openai, anthropic, ollama)';

