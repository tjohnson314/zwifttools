# ZwiftTools - Bike Comparison App

A web app that lets you analyze your Zwift race rides and compare how you would
have performed with a different bike/wheel setup, using Zwift's physics model.

## Features

- **OAuth Login**: Authenticate with your Zwift account securely
- **Activity Analysis**: Load any Zwift activity by URL or ID
- **Bike Comparison**: Compare your actual ride against any frame/wheel combo or against a different rider height and weight
- **Physics Model**: Uses Zwift's CdA, Crr, weight, and gradient data
- **Detailed Charts**: Watt differences across your ride

## Live Site

🌐 https://zwifttools.azurewebsites.net

## Local Development

1. Clone and install:
   ```bash
   git clone https://github.com/YOUR_USERNAME/zwifttools.git
   cd zwifttools
   pip install -r requirements.txt
   ```

2. Run:
   ```bash
   python app.py
   ```
   Open http://localhost:5000

## Deployment

This app is deployed to **Azure App Service** via GitHub Actions.

### Azure Setup (one-time)

1. Create a Resource Group and App Service in the Azure Portal
2. Download the **Publish Profile** from the App Service
3. Add these GitHub repo secrets:
   - `AZURE_WEBAPP_NAME` — your app service name (e.g., `zwifttools`)
   - `AZURE_WEBAPP_PUBLISH_PROFILE` — the full XML publish profile

Pushes to `main` trigger automatic deployment.

## Tech Stack

- **Backend**: Flask + Gunicorn
- **Frontend**: Vanilla JS + Chart.js
- **Data**: Copied from ZwifterBikes bike database
- **Auth**: Zwift OAuth 2.0
- **Hosting**: Azure App Service (Linux, Python 3.12)