"""Service for securely storing and retrieving provider credentials."""

from datetime import datetime
from typing import Dict, List

from sqlalchemy.orm import Session

from database.models import ProviderCredential
from statmate.api.security import decrypt_secret, encrypt_secret


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
    def upsert_credentials(db: Session, user_id: str, credentials: Dict[str, str]) -> List[str]:
        """Encrypt and persist credentials; one per provider per user."""
        configured: List[str] = []
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
                existing.updated_at = datetime.utcnow()
            else:
                db.add(
                    ProviderCredential(
                        user_id=user_id,
                        provider=provider,
                        encrypted_key=encrypted,
                    )
                )
            configured.append(provider)
        db.commit()
        return configured

    @staticmethod
    def load_credentials(db: Session, user_id: str) -> Dict[str, str]:
        """Return decrypted credentials for a user keyed by provider."""
        records = db.query(ProviderCredential).filter(ProviderCredential.user_id == user_id).all()
        return {rec.provider: decrypt_secret(rec.encrypted_key) for rec in records}
