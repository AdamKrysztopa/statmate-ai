-- Migration: Add versioning and comments to analyses table
-- Date: 2025-02-20
-- Description: Track analysis versions, supersession timestamps, and user comments

ALTER TABLE analyses ADD COLUMN version INTEGER DEFAULT 1;
ALTER TABLE analyses ADD COLUMN superseded_at TIMESTAMP NULL;
ALTER TABLE analyses ADD COLUMN comment TEXT;

UPDATE analyses SET version = COALESCE(version, 1) WHERE version IS NULL;
