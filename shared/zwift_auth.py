"""
Zwift OAuth Authentication Module.

Provides OAuth 2.0 Authorization Code Flow for Zwift API authentication.
Users are redirected to Zwift's login page and back to your app with tokens.
"""

import requests
import time
from dataclasses import dataclass


# Zwift OAuth endpoints
ZWIFT_AUTH_URL = "https://secure.zwift.com/auth/realms/zwift/protocol/openid-connect/auth"
ZWIFT_TOKEN_URL = "https://secure.zwift.com/auth/realms/zwift/protocol/openid-connect/token"

# Public client ID used by Zwift Companion app
# This is not a secret - it's embedded in the app
CLIENT_ID = "Zwift_Mobile_Link"


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
    
    resp = requests.post(ZWIFT_TOKEN_URL, data=data)
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
    
    resp = requests.post(ZWIFT_TOKEN_URL, data=data)
    resp.raise_for_status()
    
    token_data = resp.json()
    expires_at = time.time() + token_data.get('expires_in', 3600)
    
    return ZwiftTokens(
        access_token=token_data['access_token'],
        refresh_token=token_data['refresh_token'],
        expires_at=expires_at
    )
