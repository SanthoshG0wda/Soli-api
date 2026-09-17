# PowerShell script to install pgvector for PostgreSQL 18 on Windows
param (
    [string]$PgRoot = "C:\Program Files\PostgreSQL\18"
)

$ErrorActionPreference = "Stop"
Write-Host "Installing pgvector for PostgreSQL 18..." -ForegroundColor Cyan

$zipUrl = "https://github.com/andreiramani/pgvector_pgsql_windows/releases/download/0.8.6_18/vector.v0.8.6-pg18.zip"
$tempZip = Join-Path $env:TEMP "vector.v0.8.6-pg18.zip"
$tempExtract = Join-Path $env:TEMP "pgvector_extracted"

Write-Host "Downloading pgvector v0.8.6 from GitHub..."
Invoke-WebRequest -Uri $zipUrl -OutFile $tempZip -UseBasicParsing

if (Test-Path $tempExtract) { Remove-Item -Recurse -Force $tempExtract }
Expand-Archive -Path $tempZip -DestinationPath $tempExtract -Force

Write-Host "Copying files to $PgRoot..."
Copy-Item -Path "$tempExtract\lib\vector.dll" -Destination "$PgRoot\lib\" -Force
Copy-Item -Path "$tempExtract\share\extension\*" -Destination "$PgRoot\share\extension\" -Force

if (-not (Test-Path "$PgRoot\include\server\extension\vector")) {
    New-Item -ItemType Directory -Path "$PgRoot\include\server\extension\vector" -Force | Out-Null
}
Copy-Item -Path "$tempExtract\include\server\extension\vector\*" -Destination "$PgRoot\include\server\extension\vector\" -Force

Write-Host "pgvector files copied successfully!" -ForegroundColor Green

# Clean up
Remove-Item -Force $tempZip -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force $tempExtract -ErrorAction SilentlyContinue

Write-Host "Enabling vector extension in PostgreSQL..."
& "$PgRoot\bin\psql.exe" -U postgres -d soli_db -c "CREATE EXTENSION IF NOT EXISTS vector;"

Write-Host "pgvector installation complete!" -ForegroundColor Green
