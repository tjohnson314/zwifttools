"""Gunicorn configuration for Azure App Service."""

bind = "0.0.0.0:8000"
workers = 2
threads = 4
timeout = 600  # Long timeout for Zwift API calls
accesslog = "-"
errorlog = "-"
loglevel = "info"
