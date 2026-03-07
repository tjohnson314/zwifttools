<#
.SYNOPSIS
    Deploys the Zwift Games Analytics infrastructure end-to-end.

.DESCRIPTION
    1. Deploys the Bicep template (Function App, Storage, App Insights)
    2. Creates the ZwiftGames database in your Azure Data Explorer cluster
    3. Executes the KQL table-creation commands
    4. Updates the Function App settings with the Kusto connection string

.PARAMETER ResourceGroupName
    Azure resource group (must already exist). Default: zwifttools

.PARAMETER Location
    Azure region. Default: eastus2

.PARAMETER KustoClusterUri
    The query URI of your ADX cluster.

.PARAMETER KustoIngestUri
    The ingestion URI of your ADX cluster.

.PARAMETER KustoDatabaseName
    Name of the KQL database to create. Default: ZwiftGames

.PARAMETER ParametersFile
    Path to the Bicep parameters file. Default: parameters.json

.PARAMETER SkipBicep
    Skip the Bicep deployment (useful when re-running just the KQL steps).

.EXAMPLE
    .\deploy.ps1
#>

[CmdletBinding()]
param(
    [string]$ResourceGroupName = "zwifttools",
    [string]$Location = "eastus2",
    [string]$KustoClusterUri = "https://kvcayg2gmw1snvk9sceve0.southcentralus.kusto.windows.net",
    [string]$KustoIngestUri = "https://ingest-kvcayg2gmw1snvk9sceve0.southcentralus.kusto.windows.net",
    [string]$KustoDatabaseName = "ZwiftGames",
    [string]$ParametersFile = "$PSScriptRoot\parameters.json",
    [switch]$SkipBicep
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# -- Helpers -----------------------------------------------------------------

function Write-Step { param([string]$msg) Write-Host "`n> $msg" -ForegroundColor Cyan }
function Write-Ok { param([string]$msg) Write-Host "  [OK] $msg" -ForegroundColor Green }
function Write-Warn { param([string]$msg) Write-Host "  [WARN] $msg" -ForegroundColor Yellow }

function Get-KustoAccessToken {
    $token = az account get-access-token --resource "https://kusto.kusto.windows.net" --query accessToken -o tsv
    if (-not $token) { throw "Failed to get Kusto access token. Run 'az login' first." }
    return $token
}

function Invoke-KqlCommand {
    param(
        [string]$ClusterUri,
        [string]$Database,
        [string]$Command,
        [string]$Token
    )
    $body = @{
        db  = $Database
        csl = $Command
    } | ConvertTo-Json

    $headers = @{
        "Authorization" = "Bearer $Token"
        "Content-Type"  = "application/json"
    }

    $response = Invoke-RestMethod -Method Post -Uri "$ClusterUri/v1/rest/mgmt" -Headers $headers -Body $body
    return $response
}


# ==========================================================================
# STEP 1 -- Deploy Bicep (Azure resources)
# ==========================================================================

if (-not $SkipBicep) {
    Write-Step "Deploying Bicep template to resource group '$ResourceGroupName'..."

    # Ensure resource group exists
    $rgExists = az group exists --name $ResourceGroupName
    if ($rgExists -eq "false") {
        az group create --name $ResourceGroupName --location $Location | Out-Null
        Write-Ok "Created resource group '$ResourceGroupName' in '$Location'"
    }

    $deployJson = az deployment group create `
        --resource-group $ResourceGroupName `
        --template-file "$PSScriptRoot\main.bicep" `
        --parameters "@$ParametersFile" `
        -o json

    if (-not $deployJson) { throw "Bicep deployment returned no output." }

    $deployObj = $deployJson | ConvertFrom-Json
    $outputs = $deployObj.properties.outputs

    if (-not $outputs) {
        Write-Warn "Deployment outputs not found at .properties.outputs, dumping structure:"
        Write-Host ($deployJson | Out-String)
        throw "Could not find deployment outputs."
    }

    $functionAppName = $outputs.functionAppName.value
    Write-Ok "Bicep deployment complete"
    Write-Ok "  Function App : $functionAppName"
}
else {
    Write-Warn "Skipping Bicep deployment (--SkipBicep)"
    $deployJson = az deployment group list `
        --resource-group $ResourceGroupName `
        --query "[0].properties.outputs" `
        -o json

    if (-not $deployJson) {
        throw "No previous deployment found. Run without -SkipBicep first."
    }

    $outputs = $deployJson | ConvertFrom-Json
    $functionAppName = $outputs.functionAppName.value
}


# ==========================================================================
# STEP 2 -- Create KQL database in ADX cluster
# ==========================================================================

Write-Step "Ensuring KQL database '$KustoDatabaseName' exists in ADX cluster..."

$kustoToken = Get-KustoAccessToken

# Check if database already exists by listing databases
$dbExists = $false
try {
    $dbList = Invoke-KqlCommand -ClusterUri $KustoClusterUri -Database "NetDefaultDB" -Command ".show databases" -Token $kustoToken
    foreach ($table in $dbList.Tables) {
        foreach ($row in $table.Rows) {
            # DatabaseName is typically the first column
            if ($row[0] -eq $KustoDatabaseName) {
                $dbExists = $true
                break
            }
        }
    }
}
catch {
    Write-Warn "Could not list databases: $_"
}

if ($dbExists) {
    Write-Ok "Database '$KustoDatabaseName' already exists"
}
else {
    try {
        Invoke-KqlCommand -ClusterUri $KustoClusterUri -Database "NetDefaultDB" -Command ".create database $KustoDatabaseName" -Token $kustoToken | Out-Null
        Write-Ok "Created database '$KustoDatabaseName'"
    }
    catch {
        Write-Warn "Could not create database automatically: $_"
        Write-Host ""
        Write-Host "  For free ADX clusters, you may need to create the database manually:" -ForegroundColor Yellow
        Write-Host "    1. Go to https://dataexplorer.azure.com"
        Write-Host "    2. Click 'Add cluster' and enter: $KustoClusterUri"
        Write-Host "    3. Right-click the cluster -> 'Create database' -> name it '$KustoDatabaseName'"
        Write-Host "    4. Re-run this script with: .\deploy.ps1 -SkipBicep"
        Write-Host ""
        return
    }
}


# ==========================================================================
# STEP 3 -- Execute KQL table-creation commands
# ==========================================================================

Write-Step "Creating KQL tables and policies..."

$kqlFile = Get-Content "$PSScriptRoot\kql\create_tables.kql" -Raw

# Split on double-newline, strip comments, and filter for blocks starting with '.'
$commands = @()
foreach ($block in ($kqlFile -split '\r?\n\r?\n')) {
    $lines = ($block -split '\r?\n') | Where-Object { $_ -notmatch '^\s*//' -and $_.Trim() -ne '' }
    $cleaned = ($lines -join "`n").Trim()
    if ($cleaned -and $cleaned[0] -eq '.') { $commands += $cleaned }
}

$succeeded = 0
$failed = 0

foreach ($cmd in $commands) {
    if ([string]::IsNullOrWhiteSpace($cmd)) { continue }

    $shortCmd = ($cmd -split '\r?\n')[0].Substring(0, [Math]::Min(80, ($cmd -split '\r?\n')[0].Length))
    try {
        Invoke-KqlCommand -ClusterUri $KustoClusterUri -Database $KustoDatabaseName -Command $cmd -Token $kustoToken | Out-Null
        Write-Ok $shortCmd
        $succeeded++
    }
    catch {
        Write-Warn "FAILED: $shortCmd - $_"
        $failed++
    }
}

Write-Host "  KQL commands: $succeeded succeeded, $failed failed" -ForegroundColor $(if ($failed -gt 0) { "Yellow" } else { "Green" })


# ==========================================================================
# STEP 4 -- Update Function App settings with Kusto connection info
# ==========================================================================

Write-Step "Updating Function App settings with Kusto connection..."

# Gather Zwift credentials (re-use existing values if already set)
$existingZwiftUser = az functionapp config appsettings list `
    --resource-group $ResourceGroupName --name $functionAppName `
    --query "[?name=='ZWIFT_USERNAME'].value | [0]" -o tsv 2>$null

if ($existingZwiftUser) {
    Write-Host "  ZWIFT_USERNAME is already configured ($existingZwiftUser)" -ForegroundColor DarkGray
    $zwiftUser = $existingZwiftUser
}
else {
    $zwiftUser = Read-Host "  Enter your Zwift username (email)"
}

$existingZwiftPass = az functionapp config appsettings list `
    --resource-group $ResourceGroupName --name $functionAppName `
    --query "[?name=='ZWIFT_PASSWORD'].value | [0]" -o tsv 2>$null

if ($existingZwiftPass) {
    Write-Host "  ZWIFT_PASSWORD is already configured (hidden)" -ForegroundColor DarkGray
    $zwiftPass = $existingZwiftPass
}
else {
    $zwiftPass = Read-Host "  Enter your Zwift password" -AsSecureString
    $zwiftPass = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [Runtime.InteropServices.Marshal]::SecureStringToBSTR($zwiftPass))
}

az functionapp config appsettings set `
    --resource-group $ResourceGroupName `
    --name $functionAppName `
    --settings "KUSTO_CLUSTER_URI=$KustoClusterUri" `
    "KUSTO_INGEST_URI=$KustoIngestUri" `
    "KUSTO_DATABASE=$KustoDatabaseName" `
    "ZWIFT_USERNAME=$zwiftUser" `
    "ZWIFT_PASSWORD=$zwiftPass" `
    "SCM_DO_BUILD_DURING_DEPLOYMENT=true" `
    --output none

Write-Ok "Function App settings updated (Kusto + Zwift credentials)"


# ==========================================================================
# Summary
# ==========================================================================

Write-Host "`n==================================================" -ForegroundColor Cyan
Write-Host "  Deployment complete!" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Resource Group : $ResourceGroupName"
Write-Host "  ADX Cluster    : $KustoClusterUri"
Write-Host "  KQL Database   : $KustoDatabaseName"
Write-Host "  Function App   : $functionAppName"
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor Yellow
Write-Host "    1. Deploy the function code:  func azure functionapp publish $functionAppName --build remote"
Write-Host "    2. Verify tables at https://dataexplorer.azure.com -> $KustoDatabaseName"
Write-Host ""
