$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$upstreamDirectory = Join-Path $projectRoot 'official_ddrm'
if (-not (Test-Path -LiteralPath $upstreamDirectory)) {
    git clone https://github.com/bahjat-kawar/ddrm.git $upstreamDirectory
    if ($LASTEXITCODE -ne 0) { throw 'git clone failed' }
}
git -C $upstreamDirectory checkout 32b6b3ccfda532ba01c9cc5b6e7456c3a06a6ca2
if ($LASTEXITCODE -ne 0) { throw 'git checkout failed' }
