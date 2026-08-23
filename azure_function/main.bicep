// --------------------------------------------------------------------------
// Zwift Games Analytics -- Azure Infrastructure
//
// Deploys:
//   1. Azure Function App -- runs the ETL timer trigger
//   2. Storage Account -- required by the Function App
//   3. Application Insights -- monitoring for the Function App
//   4. Log Analytics workspace -- backing store for App Insights
//
// NOTE: ADX free cluster is provisioned separately via the Azure portal.
//       The deploy.ps1 script creates the KQL database and tables.
// --------------------------------------------------------------------------

@description('Azure region for all resources')
param location string = resourceGroup().location

@description('Azure region for the Function App (use a different region if quota is exhausted in the primary region)')
param functionLocation string = location

@description('Name of an existing App Service plan to host the Function App')
param existingAppServicePlanName string

@description('Name prefix used for all resources')
param namePrefix string = 'zwiftgames'

// --------------------------------------------------------------------------
// Variables
// --------------------------------------------------------------------------

var uniqueSuffix = uniqueString(resourceGroup().id)
var storageAccountName = 'zwiftgames'
var appInsightsName = '${namePrefix}-insights'
var logAnalyticsName = '${namePrefix}-logs'
var functionAppName = '${namePrefix}-func-${uniqueSuffix}'

// --------------------------------------------------------------------------
// 1. Storage Account (for Function App)
// --------------------------------------------------------------------------

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageAccountName
  location: functionLocation
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
  }
}

// --------------------------------------------------------------------------
// 3. Log Analytics + Application Insights
// --------------------------------------------------------------------------

resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: logAnalyticsName
  location: functionLocation
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsName
  location: functionLocation
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalyticsWorkspace.id
  }
}

// --------------------------------------------------------------------------
// 4. Function App (Python 3.11, Linux) on existing App Service Plan
// --------------------------------------------------------------------------

// Reference the existing App Service plan in this resource group
resource existingPlan 'Microsoft.Web/serverfarms@2023-12-01' existing = {
  name: existingAppServicePlanName
}

resource functionApp 'Microsoft.Web/sites@2023-12-01' = {
  name: functionAppName
  location: functionLocation
  kind: 'functionapp,linux'
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    serverFarmId: existingPlan.id
    httpsOnly: true
    siteConfig: {
      linuxFxVersion: 'Python|3.11'
      appSettings: [
        {
          name: 'AzureWebJobsStorage'
          value: 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};EndpointSuffix=${environment().suffixes.storage};AccountKey=${storageAccount.listKeys().keys[0].value}'
        }
        {
          name: 'FUNCTIONS_EXTENSION_VERSION'
          value: '~4'
        }
        {
          name: 'FUNCTIONS_WORKER_RUNTIME'
          value: 'python'
        }
        {
          name: 'APPINSIGHTS_INSTRUMENTATIONKEY'
          value: appInsights.properties.InstrumentationKey
        }
        {
          name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
          value: appInsights.properties.ConnectionString
        }
        // ZWIFT_USERNAME, ZWIFT_PASSWORD, KUSTO_CLUSTER_URI, KUSTO_INGEST_URI,
        // KUSTO_DATABASE, and SCM_DO_BUILD_DURING_DEPLOYMENT are set via
        // 'az functionapp config appsettings set' in deploy.ps1 so they
        // survive Bicep re-deployments (ARM replaces the entire appSettings
        // array on each deployment).
      ]
    }
  }
}

// --------------------------------------------------------------------------
// Outputs
// --------------------------------------------------------------------------

output functionAppName string = functionApp.name
output functionAppDefaultHostName string = functionApp.properties.defaultHostName
output storageAccountName string = storageAccount.name
output appInsightsName string = appInsights.name
