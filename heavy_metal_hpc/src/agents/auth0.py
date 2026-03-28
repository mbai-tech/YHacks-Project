"""Auth0 helpers for user-scoped AI-agent access."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from urllib.parse import urlencode
from time import sleep

import requests


class Auth0ConfigurationError(RuntimeError):
    """Raised when Auth0 configuration is incomplete."""


class Auth0AuthorizationPending(RuntimeError):
    """Raised while waiting for the user to complete device authorization."""


@dataclass
class Auth0AgentContext:
    """Small helper for building authenticated requests for user-scoped agents.

    This does not attempt to fully implement an OAuth browser flow. Instead it
    centralizes the configuration an agent needs once it already has a user or
    exchanged access token from Auth0.
    """

    domain: str
    audience: str
    client_id: str
    client_secret: str
    token_vault_audience: str | None = None

    @classmethod
    def from_env(
        cls,
        domain: str,
        audience: str,
        client_id_env: str = "AUTH0_CLIENT_ID",
        client_secret_env: str = "AUTH0_CLIENT_SECRET",
        token_vault_audience: str | None = None,
    ) -> "Auth0AgentContext":
        """Build a context using environment-provided credentials."""
        client_id = os.getenv(client_id_env)
        client_secret = os.getenv(client_secret_env)
        if not client_id or not client_secret:
            raise Auth0ConfigurationError(
                "Missing Auth0 client credentials. "
                f"Expected {client_id_env} and {client_secret_env}."
            )
        return cls(
            domain=domain,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret,
            token_vault_audience=token_vault_audience,
        )

    def token_endpoint(self) -> str:
        """Return the OAuth token endpoint."""
        return f"https://{self.domain}/oauth/token"

    def client_credentials_payload(self) -> dict[str, str]:
        """Return the payload for machine-to-machine token exchange."""
        return {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "audience": self.audience,
        }

    def user_api_headers(self, access_token: str) -> dict[str, str]:
        """Build headers for calling first-party APIs on behalf of a user."""
        return {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

    def describe_agent_scope(self) -> dict[str, str | None]:
        """Return a serializable summary for agent startup."""
        return {
            "auth0_domain": self.domain,
            "audience": self.audience,
            "token_vault_audience": self.token_vault_audience,
        }

    def device_authorization_endpoint(self) -> str:
        """Return the device authorization endpoint."""
        return f"https://{self.domain}/oauth/device/code"

    def start_device_flow(self, scope: str = "openid profile email") -> dict[str, str | int]:
        """Start an OAuth device authorization flow."""
        response = requests.post(
            self.device_authorization_endpoint(),
            data={
                "client_id": self.client_id,
                "scope": scope,
                "audience": self.audience,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()

    def poll_device_token(
        self,
        device_code: str,
        interval: int = 5,
        timeout_s: int = 300,
    ) -> dict[str, str]:
        """Poll Auth0 until the user completes authorization."""
        waited = 0
        while waited <= timeout_s:
            response = requests.post(
                self.token_endpoint(),
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": device_code,
                    "client_id": self.client_id,
                },
                timeout=30.0,
            )
            data = response.json()
            if response.ok and "access_token" in data:
                return data

            error = data.get("error")
            if error in {"authorization_pending", "slow_down"}:
                if timeout_s <= interval:
                    raise TimeoutError("Authorization still pending.")
                sleep(interval + (5 if error == "slow_down" else 0))
                waited += interval
                continue

            raise Auth0ConfigurationError(f"Auth0 device flow failed: {data}")

        raise TimeoutError("Timed out waiting for Auth0 device authorization.")

    def fetch_user_profile(self, access_token: str) -> dict[str, str]:
        """Fetch the authenticated user's profile from Auth0."""
        response = requests.get(
            f"https://{self.domain}/userinfo",
            headers=self.user_api_headers(access_token),
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()

    def build_authorize_url(
        self,
        redirect_uri: str,
        scope: str = "openid profile email",
        state: str | None = None,
        include_audience: bool = True,
    ) -> tuple[str, str]:
        """Build an Auth0 authorization URL for browser login."""
        actual_state = state or secrets.token_urlsafe(24)
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "scope": scope,
            "state": actual_state,
        }
        if include_audience and self.audience:
            params["audience"] = self.audience
        query = urlencode(params)
        return f"https://{self.domain}/authorize?{query}", actual_state

    def exchange_authorization_code(
        self,
        code: str,
        redirect_uri: str,
    ) -> dict[str, str]:
        """Exchange an authorization code for tokens."""
        response = requests.post(
            self.token_endpoint(),
            json={
                "grant_type": "authorization_code",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "code": code,
                "redirect_uri": redirect_uri,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()

    def build_logout_url(self, return_to: str) -> str:
        """Build an Auth0 logout URL that also clears the upstream SSO session."""
        query = urlencode(
            {
                "client_id": self.client_id,
                "returnTo": return_to,
            }
        )
        return f"https://{self.domain}/v2/logout?{query}"
