[CmdletBinding()]
param(
    [string]$BuildDirectory = $PSScriptRoot,
    [string]$OutputDirectory = (Join-Path $PSScriptRoot 'release'),
    [string]$SevenZipPath = (Join-Path $env:ProgramFiles '7-Zip\7z.exe'),
    [ValidateRange(1, 64)]
    [int]$Threads = 16
)

$ErrorActionPreference = 'Stop'
$buildPath = (Resolve-Path -LiteralPath $BuildDirectory).Path
if (-not (Test-Path -LiteralPath $SevenZipPath -PathType Leaf)) {
    $sevenZipCommand = Get-Command 7z.exe -ErrorAction SilentlyContinue
    if (-not $sevenZipCommand) { throw 'Install 7-Zip or supply -SevenZipPath.' }
    $SevenZipPath = $sevenZipCommand.Source
}
foreach ($required in @('FrequencyCleaner-GPU.exe', '_internal_gpu\torch\lib\torch_cuda.dll')) {
    if (-not (Test-Path -LiteralPath (Join-Path $buildPath $required) -PathType Leaf)) {
        throw "Missing GPU build file: $required. Run build_gpu.bat first."
    }
}
foreach ($document in @('README.md', 'LICENSE')) {
    $source = Join-Path $PSScriptRoot $document
    $destination = Join-Path $buildPath $document
    if ($source -ne $destination) {
        Copy-Item -LiteralPath $source -Destination $destination -Force
    }
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$outputPath = (Resolve-Path -LiteralPath $OutputDirectory).Path
$archiveName = 'FrequencyCleaner-GPU.7z'
$archivePath = Join-Path $outputPath $archiveName
$temporaryArchive = Join-Path $outputPath ('.gpu-' + [guid]::NewGuid().ToString('N') + '.partial.7z')

# Explicit inputs keep tracks, presets, history, and development files out.
# A large solid dictionary helps the bundled CUDA DLLs fit in a single asset.
Push-Location -LiteralPath $buildPath
try {
    & $SevenZipPath a -t7z -mx=9 -m0=LZMA2 -md=256m -mfb=273 -ms=on "-mmt=$Threads" -bsp0 `
        $temporaryArchive 'FrequencyCleaner-GPU.exe' '_internal_gpu' 'README.md' 'LICENSE'
    if ($LASTEXITCODE -ne 0) { throw "7-Zip compression failed ($LASTEXITCODE)." }
} finally {
    Pop-Location
}

$archiveBytes = (Get-Item -LiteralPath $temporaryArchive).Length
if ($archiveBytes -ge 2GB) {
    throw "Archive is $archiveBytes bytes, above GitHub's 2 GiB asset limit. Kept at $temporaryArchive."
}
& $SevenZipPath t -bsp0 $temporaryArchive
if ($LASTEXITCODE -ne 0) { throw "7-Zip integrity verification failed ($LASTEXITCODE)." }

Move-Item -LiteralPath $temporaryArchive -Destination $archivePath -Force
$digest = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
$checksumPath = $archivePath + '.sha256'
Set-Content -LiteralPath $checksumPath -Value "$digest  $archiveName" -Encoding Ascii

Write-Output "Archive: $archivePath"
Write-Output ('Size: {0:N0} bytes ({1:N3} GiB), below the 2 GiB release limit' -f $archiveBytes, ($archiveBytes / 1GB))
Write-Output "SHA256: $digest"
Write-Output "Checksum: $checksumPath"
