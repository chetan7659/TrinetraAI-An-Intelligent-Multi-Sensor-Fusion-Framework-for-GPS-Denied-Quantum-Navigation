# ==============================================================================
# Trinetra-AI — Environment Setup (Windows PowerShell)
# ==============================================================================
# Creates a virtual environment, activates it, and installs dependencies.
#
# Usage:
#   .\scripts\setup_environment.ps1
# ==============================================================================

$ErrorActionPreference = "Stop"

$VenvDir = ".venv"
$Requirements = "requirements.txt"

Write-Host "============================================================"
Write-Host "  Trinetra-AI — Environment Setup"
Write-Host "============================================================"

# ── 1. Check Python version ──────────────────────────
$PythonCmd = $null
foreach ($cmd in @("python3", "python")) {
    try {
        $version = & $cmd --version 2>&1
        if ($version -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -ge 3 -and $minor -ge 11) {
                $PythonCmd = $cmd
                Write-Host "  Using: $cmd ($version)"
                break
            }
        }
    } catch {
        continue
    }
}

if (-not $PythonCmd) {
    Write-Host "  ERROR: Python 3.11+ not found. Please install Python >= 3.11." -ForegroundColor Red
    exit 1
}

# ── 2. Create virtual environment ────────────────────
if (Test-Path $VenvDir) {
    Write-Host "  Virtual environment already exists at $VenvDir"
} else {
    Write-Host "  Creating virtual environment at $VenvDir ..."
    & $PythonCmd -m venv $VenvDir
}

# ── 3. Activate virtual environment ──────────────────
Write-Host "  Activating virtual environment ..."
$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
if (Test-Path $ActivateScript) {
    & $ActivateScript
} else {
    Write-Host "  ERROR: Activation script not found at $ActivateScript" -ForegroundColor Red
    exit 1
}

# ── 4. Upgrade pip ───────────────────────────────────
Write-Host "  Upgrading pip ..."
pip install --upgrade pip --quiet

# ── 5. Install dependencies ─────────────────────────
if (Test-Path $Requirements) {
    Write-Host "  Installing dependencies from $Requirements ..."
    pip install -r $Requirements --quiet
} else {
    Write-Host "  WARNING: $Requirements not found. Skipping dependency install." -ForegroundColor Yellow
}

# ── 6. Summary ───────────────────────────────────────
Write-Host "============================================================"
Write-Host "  ✅ Environment ready."
Write-Host ""
Write-Host "  Activate later with:"
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "  Verify with:"
Write-Host "    python scripts\check_environment.py"
Write-Host "============================================================"
