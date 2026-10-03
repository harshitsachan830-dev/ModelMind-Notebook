param(
    [string]$Port = "8899",
    [string]$Workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")),
    [string]$PythonExe = "py"
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venv = Join-Path $root ".venv"
$envFile = Join-Path $root ".env"

if (Test-Path $envFile) {
    $allowedSettings = @("ML_PLATFORM_CLOUD_FALLBACK", "GEMINI_API_KEY", "GEMINI_MODEL")
    foreach ($line in Get-Content $envFile) {
        $entry = $line.Trim()
        if (-not $entry -or $entry.StartsWith("#")) {
            continue
        }

        $parts = $entry -split "=", 2
        if ($parts.Count -ne 2) {
            throw "Invalid .env entry. Use KEY=VALUE format."
        }

        $name = $parts[0].Trim()
        if ($name -notin $allowedSettings) {
            throw "Unsupported .env setting: $name"
        }

        $value = $parts[1].Trim()
        if ($value.Length -ge 2 -and (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'")))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
    Write-Host "Loaded Gemini settings from .env (values hidden)."
}

if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
    Write-Host "Creating local virtual environment at $venv"
    & $PythonExe -m venv $venv
}

$python = Join-Path $venv "Scripts\python.exe"
$jupyter = Join-Path $venv "Scripts\jupyter.exe"

& $python -m pip install --upgrade pip
& $python -m pip install -r (Join-Path $root "requirements.txt")
& $python -m pip install -e (Join-Path $root "extensions\error-assistant")

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama is not installed. The local AI route is optional; install it from https://ollama.com/download"
    Write-Host "Model installation remains explicit and must be approved before any pull command."
}

& $jupyter lab --no-browser --port $Port --ServerApp.root_dir $Workspace
