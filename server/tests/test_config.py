"""Deployment config: what Render passes in must be read correctly."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from app.config import Settings
from app.evolution.signing import ManifestSigner
from app.main import make_signer


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, supabase_url="https://x.supabase.co", supabase_service_role_key=SecretStr("k"),
                    admin_token=SecretStr("strong-token"), **kw)


@pytest.mark.parametrize("variant", ["multiline", "escaped_one_line", "crlf"])
def test_ota_key_from_env_in_any_paste_form(variant):
    signer = ManifestSigner.generate()
    pem = signer.private_pem().decode()
    value = {"multiline": pem, "escaped_one_line": pem.strip().replace("\n", "\\n"),
             "crlf": pem.replace("\n", "\r\n")}[variant]
    loaded = make_signer(_settings(ota_signing_key_pem=SecretStr(value)))
    assert loaded.public_pem() == signer.public_pem()


def test_production_guards():
    with pytest.raises(ValueError, match="ADMIN_TOKEN"):
        Settings(_env_file=None, supabase_url="https://x.supabase.co", supabase_service_role_key=SecretStr("k"))
    with pytest.raises(ValueError, match="OTA_SIGNING_KEY_PEM"):
        make_signer(_settings())
