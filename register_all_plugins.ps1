<#
.SYNOPSIS
    Registers all 3 AI plugins (mrRemBG, mrAIGen3D, mrGenSfM) into AliceVision Meshroom 2025.1.0+
    via Windows Directory Junctions and copies official pipeline templates.
#>
param(
    [string]$MeshroomRoot = "D:\Meshroom\Meshroom-2025.1.0"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$Plugins = @(
    @{ Name = "mrRemBG";   Path = Join-Path $ScriptDir "mrRemBG" },
    @{ Name = "mrAIGen3D"; Path = Join-Path $ScriptDir "mrAIGen3D" },
    @{ Name = "mrGenSfM";  Path = Join-Path $ScriptDir "mrGenSfM" }
)

$MeshroomPluginsDir = Join-Path $MeshroomRoot "plugins"
if (-not (Test-Path $MeshroomPluginsDir)) {
    New-Item -ItemType Directory -Path $MeshroomPluginsDir -Force | Out-Null
}

$ShareMeshroomDir = Join-Path $MeshroomRoot "aliceVision\share\meshroom"
if (-not (Test-Path $ShareMeshroomDir)) {
    New-Item -ItemType Directory -Path $ShareMeshroomDir -Force | Out-Null
}

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host " Registering AliceVision Meshroom AI Add-ons Suite " -ForegroundColor Cyan
Write-Host " Target Meshroom: $MeshroomRoot" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan

foreach ($plugin in $Plugins) {
    $pName = $plugin.Name
    $pSrc = $plugin.Path
    $targetJunction = Join-Path $MeshroomPluginsDir $pName

    Write-Host "`n[+] Processing $pName..." -ForegroundColor Yellow

    if (Test-Path $targetJunction) {
        $item = Get-Item $targetJunction -Force
        if ($item.LinkType -eq "Junction" -or $item.LinkType -eq "SymbolicLink") {
            Write-Host "    Junction already exists: $targetJunction -> $($item.Target)" -ForegroundColor Green
        } else {
            Write-Warning "    Path $targetJunction already exists and is not a junction. Skipping."
        }
    } else {
        New-Item -ItemType Junction -Path $targetJunction -Target $pSrc | Out-Null
        Write-Host "    Created plugin junction: $targetJunction -> $pSrc" -ForegroundColor Green
    }

    # Copy pipeline templates (.mg)
    $pipelinesDir = Join-Path $pSrc "pipelines"
    if (Test-Path $pipelinesDir) {
        Get-ChildItem -Path $pipelinesDir -Filter "*.mg" | ForEach-Object {
            Copy-Item $_.FullName -Destination (Join-Path $ShareMeshroomDir $_.Name) -Force
            Write-Host "    Installed pipeline template: $($_.Name)" -ForegroundColor Cyan
        }
    }
}

Write-Host "`n=========================================================" -ForegroundColor Green
Write-Host " All plugins registered successfully! Ready to launch Meshroom. " -ForegroundColor Green
Write-Host "=========================================================" -ForegroundColor Green
