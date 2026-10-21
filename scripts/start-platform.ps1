param(
    [string]$Port = "8899",
    [string]$Workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")),
    [string]$PythonExe = "py"
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venv = Join-Path $root ".venv"
$envFile = Join-Path $root ".env"
$configPath = Join-Path $root "jupyter-config"

$env:MODELMIND_JUPYTER_PORT = $Port

function Invoke-CheckedCommand {
    param(
        [string]$FilePath,
        [string[]]$CommandArguments
    )

    & $FilePath @CommandArguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE."
    }
}

if ($env:JUPYTER_CONFIG_PATH) {
    $env:JUPYTER_CONFIG_PATH = "$configPath;$env:JUPYTER_CONFIG_PATH"
} else {
    $env:JUPYTER_CONFIG_PATH = $configPath
}

if ($env:WORKSPACE) {
    $Workspace = $env:WORKSPACE
}
$Workspace = $Workspace.Trim().Trim('"', "'").TrimEnd('\')
if (-not (Test-Path -LiteralPath $Workspace)) {
    [System.IO.Directory]::CreateDirectory($Workspace) | Out-Null
}

if (Test-Path $envFile) {
    $allowedSettings = @("ML_PLATFORM_CLOUD_FALLBACK", "GEMINI_API_KEY", "GEMINI_MODEL", "MODELMIND_WEB_ORIGINS")
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
    if (-not (Get-Command $PythonExe -ErrorAction SilentlyContinue)) {
        throw "Python launcher '$PythonExe' was not found. Install Python or pass its command with -PythonExe."
    }
    Invoke-CheckedCommand -FilePath $PythonExe -CommandArguments @("-m", "venv", $venv)
}

$python = Join-Path $venv "Scripts\python.exe"
$jupyter = Join-Path $venv "Scripts\jupyter.exe"
$extensionLink = Join-Path $venv "share\jupyter\labextensions\@ml-platform\error-assistant"

if (-not (Test-Path $python)) {
    throw "The virtual environment Python executable was not created: $python"
}

if (-not (Test-Path $jupyter) -or $SetupOnly -or $env:MODELMIND_SETUP_ONLY -eq "1") {
    Write-Host "Installing ModelMind dependencies into virtual environment..."
    Invoke-CheckedCommand -FilePath $python -CommandArguments @("-m", "pip", "install", "--upgrade", "pip")
    Invoke-CheckedCommand -FilePath $python -CommandArguments @("-m", "pip", "install", "-r", (Join-Path $root "requirements.txt"))
    Invoke-CheckedCommand -FilePath $python -CommandArguments @("-m", "pip", "install", "-r", (Join-Path $root "modelmind-source\backend\requirements.txt"))
    if (Test-Path -LiteralPath $extensionLink) {
        $extensionItem = Get-Item -LiteralPath $extensionLink -Force
        if ($extensionItem.LinkType) {
            Remove-Item -LiteralPath $extensionLink
        }
    }
    Invoke-CheckedCommand -FilePath $python -CommandArguments @("-m", "pip", "install", (Join-Path $root "extensions\error-assistant"))
}

if (-not (Test-Path $jupyter)) {
    throw "Jupyter was not installed in the virtual environment: $jupyter"
}

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama is not installed. The local AI route is optional; install it from https://ollama.com/download"
    Write-Host "Model installation remains explicit and must be approved before any pull command."
}

& $jupyter lab --no-browser --port $Port --ServerApp.ip=127.0.0.1 --ServerApp.token='' --ServerApp.password='' --ServerApp.open_browser=False --ExtensionApp.open_browser=False --LabServerApp.open_browser=False "--ServerApp.root_dir=$Workspace"
if ($LASTEXITCODE -ne 0) {
    throw "JupyterLab exited with code $LASTEXITCODE."
}
