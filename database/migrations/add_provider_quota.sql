-- Migration: add quota fields to provider credentials
ALTER TABLE provider_credentials ADD COLUMN quota_limit INTEGER;
ALTER TABLE provider_credentials ADD COLUMN quota_used INTEGER DEFAULT 0 NOT NULL;
ALTER TABLE provider_credentials ADD COLUMN quota_reset_at DATETIME;
