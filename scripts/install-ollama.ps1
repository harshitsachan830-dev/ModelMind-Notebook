param(
    [string]$ModelId = "qwen2.5-coder:7b"
)

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Error "Ollama is not installed. Install it from https://ollama.com/download"
    exit 1
}

$installed = ollama list
if ($installed -notmatch [regex]::Escape($ModelId)) {
    Write-Host "Pulling model: $ModelId"
    ollama pull $ModelId
} else {
    Write-Host "Model already installed: $ModelId"
}

Write-Host "Model ready: $ModelId"
