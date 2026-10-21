param(
    [string]$Port = "8899",
    [string]$Workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")),
    [string]$PythonExe = "py",
    [switch]$SetupOnly
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$userDataDirectory = if ($env:MODELMIND_USER_DATA_DIR) {
    $env:MODELMIND_USER_DATA_DIR
} else {
    $root
}
$venv = if ($env:MODELMIND_VENV_DIR) {
    $env:MODELMIND_VENV_DIR
} else {
    Join-Path $root ".venv"
}
$envFile = Join-Path $userDataDirectory ".env"
$configPath = Join-Path $root "jupyter-config"
$packagedPython = Join-Path $root "runtime\python\python.exe"
if ((Test-Path $packagedPython) -and $PythonExe -eq "py") {
    $PythonExe = $packagedPython
}

$env:MODELMIND_JUPYTER_PORT = $Port

# Strip out external PYTHONHOME and ensure our venv takes strict precedence on PATH
# This prevents Anaconda or global Python from hijacking jupyter or python subcommands
Remove-Item Env:\PYTHONHOME -ErrorAction SilentlyContinue
$env:PATH = "$venv\Scripts;$venv;$env:PATH"

$serverConfigDir = Join-Path $venv "etc\jupyter\jupyter_server_config.d"
if ($env:JUPYTER_CONFIG_PATH) {
    $env:JUPYTER_CONFIG_PATH = "$configPath;$serverConfigDir;$env:JUPYTER_CONFIG_PATH"
} else {
    $env:JUPYTER_CONFIG_PATH = "$configPath;$serverConfigDir"
}

$errorAssistantDir = Join-Path $root "extensions\error-assistant"
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$errorAssistantDir;$env:PYTHONPATH"
} else {
    $env:PYTHONPATH = $errorAssistantDir
}

$venvJupyter = Join-Path $venv "share\jupyter"
$appDataJupyter = Join-Path $env:APPDATA "jupyter"
if ($env:JUPYTER_PATH) {
    $env:JUPYTER_PATH = "$venvJupyter;$appDataJupyter;$env:JUPYTER_PATH"
} else {
    $env:JUPYTER_PATH = "$venvJupyter;$appDataJupyter"
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
    New-Item -ItemType Directory -Path (Split-Path $venv) -Force | Out-Null
    & $PythonExe -m venv $venv
}

$python = Join-Path $venv "Scripts\python.exe"
$jupyter = Join-Path $venv "Scripts\jupyter.exe"
$extensionLink = Join-Path $venv "share\jupyter\labextensions\@ml-platform\error-assistant"
$builtExtensionDir = Join-Path $root "extensions\error-assistant\ml_platform_error_assistant\labextension"
$serverConfigDir = Join-Path $venv "etc\jupyter\jupyter_server_config.d"
$serverConfigFile = Join-Path $root "extensions\error-assistant\jupyter-config\server-config\ml_platform_error_assistant.json"

$isExtensionInstalled = $false
try {
    & $python -c "import ml_platform_error_assistant" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $isExtensionInstalled = $true
    }
} catch { }

if (-not (Test-Path $jupyter) -or -not (Test-Path (Join-Path $extensionLink "package.json")) -or -not $isExtensionInstalled -or $SetupOnly -or $env:MODELMIND_SETUP_ONLY -eq "1") {
    Write-Host "Installing ModelMind dependencies into virtual environment..."
    & $python -m pip install --upgrade pip
    & $python -m pip install -r (Join-Path $root "requirements.txt")
    & $python -m pip install -r (Join-Path $root "modelmind-source\backend\requirements.txt")

    $wheel = Get-ChildItem (Join-Path $root "wheels\ml_platform_error_assistant-*.whl") -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($wheel) {
        Write-Host "Installing ModelMind extension from pre-built wheel..."
        & $python -m pip install --no-deps $wheel.FullName --force-reinstall
    } else {
        & $python -m pip install --no-build-isolation (Join-Path $root "extensions\error-assistant")
    }
}

# Ensure pre-built labextension files are in place in venv and APPDATA
if (Test-Path $builtExtensionDir) {
    $locations = @(
        $extensionLink,
        (Join-Path $env:APPDATA "jupyter\labextensions\@ml-platform\error-assistant")
    )

    foreach ($loc in $locations) {
        try {
            New-Item -ItemType Directory -Path $loc -Force | Out-Null
            Copy-Item -Path (Join-Path $builtExtensionDir "*") -Destination $loc -Recurse -Force
        } catch { }
    }
}

# Ensure jupyter_server_config.d has ml_platform_error_assistant enabled across all Jupyter config directories
$serverExtensionJson = '{"ServerApp": {"jpserver_extensions": {"ml_platform_error_assistant": true}}}'
$configLocations = @(
    $serverConfigDir,
    (Join-Path $configPath "jupyter_server_config.d"),
    (Join-Path $env:APPDATA "jupyter\jupyter_server_config.d")
)

foreach ($cloc in $configLocations) {
    try {
        New-Item -ItemType Directory -Path $cloc -Force | Out-Null
        Set-Content -Path (Join-Path $cloc "ml_platform_error_assistant.json") -Value $serverExtensionJson -Force
    } catch { }
}

if ($SetupOnly -or $env:MODELMIND_SETUP_ONLY -eq "1") {
    Write-Host "ModelMind Python environment is ready."
    exit 0
}

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama is not installed. The local AI route is optional; install it from https://ollama.com/download"
    Write-Host "Model installation remains explicit and must be approved before any pull command."
}

# Use python -m jupyterlab to guarantee running inside the ModelMind virtual environment
# and never fall back to an external Python / Anaconda installation.
& $python -m jupyterlab `
    --no-browser `
    --port $Port `
    --ServerApp.ip=127.0.0.1 `
    --ServerApp.token='' `
    --ServerApp.password='' `
    --IdentityProvider.token='' `
    --IdentityProvider.password='' `
    --ServerApp.open_browser=False `
    --ExtensionApp.open_browser=False `
    --LabServerApp.open_browser=False `
    "--ServerApp.root_dir=$Workspace" `
    --ServerApp.jpserver_extensions="ml_platform_error_assistant=True"

