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
try {
    if (-not (Test-Path -LiteralPath $Workspace)) {
        [System.IO.Directory]::CreateDirectory($Workspace) | Out-Null
    }
} catch {
    $homeDir = if ($env:USERPROFILE) { $env:USERPROFILE } else { $HOME }
    $Workspace = Join-Path $homeDir "ModelMind Notebooks"
    try {
        if (-not (Test-Path -LiteralPath $Workspace)) {
            [System.IO.Directory]::CreateDirectory($Workspace) | Out-Null
        }
    } catch { }
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

$builtExtensionDir = Join-Path $root "extensions\error-assistant\ml_platform_error_assistant\labextension"
$serverConfigDir = Join-Path $venv "etc\jupyter\jupyter_server_config.d"
$serverConfigFile = Join-Path $root "extensions\error-assistant\jupyter-config\server-config\ml_platform_error_assistant.json"

if (-not (Test-Path $jupyter) -or -not (Test-Path (Join-Path $extensionLink "package.json")) -or $SetupOnly -or $env:MODELMIND_SETUP_ONLY -eq "1") {
    Write-Host "Installing ModelMind dependencies into virtual environment..."
    Invoke-CheckedCommand -FilePath $python -CommandArguments @("-m", "pip", "install", "--upgrade", "pip")
    Invoke-CheckedCommand -FilePath $python -CommandArguments @("-m", "pip", "install", "-r", (Join-Path $root "requirements.txt"))
    Invoke-CheckedCommand -FilePath $python -CommandArguments @("-m", "pip", "install", "-r", (Join-Path $root "modelmind-source\backend\requirements.txt"))

    $wheel = Get-ChildItem (Join-Path $root "wheels\ml_platform_error_assistant-*.whl") -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($wheel) {
        Write-Host "Installing ModelMind extension from pre-built wheel..."
        Invoke-CheckedCommand -FilePath $python -CommandArguments @("-m", "pip", "install", "--no-deps", $wheel.FullName, "--force-reinstall")
    } else {
        Invoke-CheckedCommand -FilePath $python -CommandArguments @("-m", "pip", "install", "--no-build-isolation", (Join-Path $root "extensions\error-assistant"))
    }

    # Ensure pre-built labextension files are in place in share/jupyter/labextensions
    if (Test-Path $builtExtensionDir) {
        New-Item -ItemType Directory -Path $extensionLink -Force | Out-Null
        Copy-Item -Path (Join-Path $builtExtensionDir "*") -Destination $extensionLink -Recurse -Force
    }

    # Ensure jupyter_server_config.d is enabled
    if (Test-Path $serverConfigFile) {
        New-Item -ItemType Directory -Path $serverConfigDir -Force | Out-Null
        Copy-Item -LiteralPath $serverConfigFile -Destination (Join-Path $serverConfigDir "ml_platform_error_assistant.json") -Force
    }
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
