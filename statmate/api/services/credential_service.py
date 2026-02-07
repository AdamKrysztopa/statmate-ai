"""Service for securely storing and retrieving provider credentials."""

from datetime import datetime

from sqlalchemy.orm import Session

from database.models import ProviderCredential
from statmate.api.security import decrypt_secret, encrypt_secret


class QuotaExceededError(Exception):
    """Raised when a user exceeds their configured quota for a provider."""


class CredentialService:
    """Handle encrypted credential persistence."""

    PROVIDER_FIELDS = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'google': 'GOOGLE_API_KEY',
        'gemini': 'GEMINI_API_KEY',
        'groq': 'GROQ_API_KEY',
        'ollama': None,  # No secret needed
    }

    @staticmethod
    def upsert_credentials(
        db: Session, user_id: str, credentials: dict[str, str], quota_limits: dict[str, int] | None = None
    ) -> list[str]:
        """Encrypt and persist credentials; one per provider per user."""
        configured: list[str] = []
        quotas = quota_limits or {}
        for provider, key in credentials.items():
            if not key:
                continue
            encrypted = encrypt_secret(key)
            existing = (
                db.query(ProviderCredential)
                .filter(ProviderCredential.user_id == user_id, ProviderCredential.provider == provider)
                .first()
            )
            if existing:
                existing.encrypted_key = encrypted
                if provider in quotas:
                    existing.quota_limit = quotas.get(provider)
                    existing.quota_used = 0
                    existing.quota_reset_at = None
                existing.updated_at = datetime.utcnow()
            else:
                db.add(
                    ProviderCredential(
                        user_id=user_id,
                        provider=provider,
                        encrypted_key=encrypted,
                        quota_limit=quotas.get(provider),
                    )
                )
            configured.append(provider)
        db.commit()
        return configured

    @staticmethod
    def update_quota_limits(db: Session, user_id: str, limits: dict[str, int]) -> None:
        """Persist quota limits without changing API keys."""
        for provider, limit in limits.items():
            rec = (
                db.query(ProviderCredential)
                .filter(ProviderCredential.user_id == user_id, ProviderCredential.provider == provider)
                .first()
            )
            if not rec:
                continue
            rec.quota_limit = limit
            rec.quota_used = 0 if limit is not None else rec.quota_used
            rec.quota_reset_at = None
        db.commit()

    @staticmethod
    def enforce_quota(db: Session, user_id: str, provider: str) -> None:
        """Check quota for a provider and increment usage."""
        rec = (
            db.query(ProviderCredential)
            .filter(ProviderCredential.user_id == user_id, ProviderCredential.provider == provider)
            .first()
        )
        if not rec or rec.quota_limit is None:
            return

        # Reset quota when window elapses (simple rolling window based on reset timestamp)
        if rec.quota_reset_at and rec.quota_reset_at < datetime.utcnow():
            rec.quota_used = 0
            rec.quota_reset_at = None

        if rec.quota_used >= rec.quota_limit:
            raise QuotaExceededError(f'Quota exceeded for {provider}. Limit={rec.quota_limit}.')

        rec.quota_used += 1
        db.commit()

    @staticmethod
    def load_credentials(db: Session, user_id: str) -> dict[str, str]:
        """Return decrypted credentials for a user keyed by provider."""
        records = db.query(ProviderCredential).filter(ProviderCredential.user_id == user_id).all()
        return {rec.provider: decrypt_secret(rec.encrypted_key) for rec in records}

    @staticmethod
    def get_quota_limits(db: Session, user_id: str) -> dict[str, int | None]:
        """Return configured quota limits keyed by provider."""
        records = db.query(ProviderCredential).filter(ProviderCredential.user_id == user_id).all()
        return {rec.provider: rec.quota_limit for rec in records if rec.quota_limit is not None}
