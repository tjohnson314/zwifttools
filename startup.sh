#!/bin/bash
# Azure App Service startup script

# Add bundled dependencies to Python path
export PYTHONPATH="/home/site/wwwroot/.python_packages/lib/site-packages:$PYTHONPATH"

gunicorn --bind=0.0.0.0:8000 --timeout 600 --workers 2 app:app
