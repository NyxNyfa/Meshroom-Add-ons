<#
.SYNOPSIS
    Registers the mrAIGen3D plugin into AliceVision Meshroom-2025.1.0 via a Windows Directory Junction.
#>
param(
    [string]$PluginSource = "D:\Meshroom\mrAIGen3D",
    [string]$MeshroomRoot = "D:\Meshroom\Meshroom-2025.1.0"
)

$ErrorActionPreference = "Stop"

$PluginsDir = Join-Path $MeshroomRoot "plugins"
if (-not (Test-Path $PluginsDir)) {
    New-Item -ItemType Directory -Path $PluginsDir -Force | Out-Null
}

$TargetLink = Join-Path $PluginsDir "mrAIGen3D"
if (Test-Path $TargetLink) {
    $Item = Get-Item $TargetLink -Force
    if ($Item.LinkType -eq "Junction" -or $Item.LinkType -eq "SymbolicLink") {
        Write-Host "[mrAIGen3D] Junction already exists at: $TargetLink -> $($Item.Target)" -ForegroundColor Green
    } else {
        throw "Path $TargetLink already exists and is not a link."
    }
} else {
    New-Item -ItemType Junction -Path $TargetLink -Target $PluginSource | Out-Null
    Write-Host "[mrAIGen3D] Registered plugin junction: $TargetLink -> $PluginSource" -ForegroundColor Green
}

$ShareMeshroomDir = Join-Path $MeshroomRoot "aliceVision\share\meshroom"
if (Test-Path $ShareMeshroomDir) {
    Get-ChildItem -Path (Join-Path $PluginSource "pipelines") -Filter "*.mg" | ForEach-Object {
        Copy-Item $_.FullName -Destination (Join-Path $ShareMeshroomDir $_.Name) -Force
        Write-Host "[mrAIGen3D] Registered pipeline template: $($_.Name)" -ForegroundColor Cyan
    }
}

