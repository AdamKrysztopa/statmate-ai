-- Migration: add assumption diagnostics storage to analyses
-- Date: 2025-02-12
-- Description: Persist assumption diagnostics to support streaming and exports

ALTER TABLE analyses ADD COLUMN assumption_log JSON;
