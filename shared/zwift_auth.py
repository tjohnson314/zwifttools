"""
Zwift OAuth Authentication Module.

Token primitives and process-local authentication for the dedicated server account.
"""

import requests
import time
import os
import threading
from dataclasses import dataclass
from .http_pacing import retry_after_seconds


# Zwift OAuth endpoints
ZWIFT_AUTH_URL = "https://secure.zwift.com/auth/realms/zwift/protocol/openid-connect/auth"
ZWIFT_TOKEN_URL = "https://secure.zwift.com/auth/realms/zwift/protocol/openid-connect/token"

# Public client ID used by Zwift Companion app
# This is not a secret - it's embedded in the app
CLIENT_ID = "Zwift_Mobile_Link"


class ZwiftAPIError(RuntimeError):
    """A safe, public-facing service or upstream data error."""

    def __init__(self, message, status_code=503, retry_after=None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class ServiceHeaders(dict):
    """Mark headers whose bearer token must be resolved again before each fetch."""

    def __init__(self, manager):
        self.manager = manager
        super().__init__(Authorization=f'Bearer {manager.access_token()}')


class ServerTokenManager:
    """Serialize token renewal across the server's request and fetch threads."""

    def __init__(self):
        self._lock = threading.Lock()
        self._tokens = None
        self._retry_at = 0.0
        self._error = None
        self._pace_lock = threading.Lock()
        self._next_request_at = 0.0

    @property
    def configured(self):
        return bool(os.environ.get('ZWIFT_USERNAME') and os.environ.get('ZWIFT_PASSWORD'))

    def access_token(self, rejected_token=None):
        with self._lock:
            if not self.configured:
                raise ZwiftAPIError('Zwift service account is not configured.')
            if time.monotonic() < self._retry_at:
                raise self._error
            tokens = self._tokens
            rejected = tokens is not None and tokens.access_token == rejected_token
            if tokens is not None and not tokens.is_expired and not rejected:
                return tokens.access_token
            try:
                if tokens is not None:
                    try:
                        tokens = refresh_access_token(tokens.refresh_token)
                    except requests.HTTPError as error:
                        if error.response is None or error.response.status_code not in (400, 401):
                            raise
                        tokens = get_token_with_password(
                            os.environ['ZWIFT_USERNAME'], os.environ['ZWIFT_PASSWORD'])
                else:
                    tokens = get_token_with_password(
                        os.environ['ZWIFT_USERNAME'], os.environ['ZWIFT_PASSWORD'])
            except (requests.RequestException, ValueError, KeyError) as error:
                self._tokens = None
                response = getattr(error, 'response', None)
                limited = response is not None and response.status_code == 429
                retry_after = response.headers.get('Retry-After') if limited else None
                self._error = ZwiftAPIError(
                    'Zwift service authentication failed.', 429 if limited else 503, retry_after)
                self._retry_at = time.monotonic() + max(30, retry_after_seconds(retry_after))
                raise self._error from None
            self._tokens = tokens
            return tokens.access_token

    def headers(self):
        return ServiceHeaders(self)

    def wait(self):
        # A single shared-account budget, including concurrent telemetry workers.
        with self._pace_lock:
            delay = self._next_request_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._next_request_at = time.monotonic() + 0.25

    def defer(self, seconds):
        with self._pace_lock:
            self._next_request_at = max(self._next_request_at, time.monotonic() + seconds)


server_token_manager = ServerTokenManager()


@dataclass
class ZwiftTokens:
    """Zwift OAuth tokens."""
    access_token: str
    refresh_token: str
    expires_at: float  # Unix timestamp when access_token expires
    
    @property
    def is_expired(self) -> bool:
        """Check if access token is expired (with 60s buffer)."""
        return time.time() >= (self.expires_at - 60)
    
    def to_dict(self) -> dict:
        return {
            'access_token': self.access_token,
            'refresh_token': self.refresh_token,
            'expires_at': self.expires_at
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ZwiftTokens':
        return cls(
            access_token=data['access_token'],
            refresh_token=data['refresh_token'],
            expires_at=data['expires_at']
        )


def exchange_code_for_tokens(code: str, redirect_uri: str) -> ZwiftTokens:
    """
    Exchange authorization code for access and refresh tokens.
    
    Args:
        code: The authorization code from the callback
        redirect_uri: Must match the redirect_uri used in the auth URL
        
    Returns:
        ZwiftTokens object with access_token, refresh_token, and expiry
        
    Raises:
        requests.HTTPError: If token exchange fails
    """
    data = {
        'client_id': CLIENT_ID,
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': redirect_uri
    }
    
    resp = requests.post(ZWIFT_TOKEN_URL, data=data)
    resp.raise_for_status()
    
    token_data = resp.json()
    expires_at = time.time() + token_data.get('expires_in', 3600)
    
    return ZwiftTokens(
        access_token=token_data['access_token'],
        refresh_token=token_data['refresh_token'],
        expires_at=expires_at
    )


def refresh_access_token(refresh_token: str) -> ZwiftTokens:
    """
    Refresh an expired access token using the refresh token.
    
    Args:
        refresh_token: The refresh token from previous authentication
        
    Returns:
        New ZwiftTokens with fresh access_token
        
    Raises:
        requests.HTTPError: If refresh fails (user may need to re-authenticate)
    """
    data = {
        'client_id': CLIENT_ID,
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token
    }
    
    resp = requests.post(ZWIFT_TOKEN_URL, data=data, timeout=30)
    resp.raise_for_status()
    
    token_data = resp.json()
    expires_at = time.time() + token_data.get('expires_in', 3600)
    
    return ZwiftTokens(
        access_token=token_data['access_token'],
        refresh_token=token_data.get('refresh_token', refresh_token),
        expires_at=expires_at
    )


def get_token_with_password(username: str, password: str) -> ZwiftTokens:
    """
    Get tokens using username/password (Resource Owner Password flow).
    
    This is useful for scripts/CLI tools where OAuth redirect isn't practical.
    For web apps, use the authorization code flow instead.
    
    Args:
        username: Zwift account email
        password: Zwift account password
        
    Returns:
        ZwiftTokens object
    """
    data = {
        'client_id': CLIENT_ID,
        'grant_type': 'password',
        'username': username,
        'password': password
    }
    
    resp = requests.post(ZWIFT_TOKEN_URL, data=data, timeout=30)
    resp.raise_for_status()
    
    token_data = resp.json()
    expires_at = time.time() + token_data.get('expires_in', 3600)
    
    return ZwiftTokens(
        access_token=token_data['access_token'],
        refresh_token=token_data['refresh_token'],
        expires_at=expires_at
    )
