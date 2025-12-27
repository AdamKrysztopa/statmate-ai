-- Migration: Add intermediate streaming fields to analyses table
-- Date: 2025-02-07
-- Description: Persist per-node decision steps and rolling intermediate logs for streaming clients

ALTER TABLE analyses ADD COLUMN decision_steps JSON;
ALTER TABLE analyses ADD COLUMN intermediate_log TEXT;

COMMENT ON COLUMN analyses.decision_steps IS 'Ordered list of agent/node decisions captured while streaming the LangGraph workflow';
COMMENT ON COLUMN analyses.intermediate_log IS 'Rolling workflow log text persisted during streaming execution';
