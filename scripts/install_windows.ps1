[CmdletBinding()]
param(
  [string]$RepoUrl = $(if ($env:SIM2REAL_REPO_URL) { $env:SIM2REAL_REPO_URL } else { "https://github.com/GuillermoRosale5/Sim2Real-MJX-JAX-sobre-Linux-WSL.git" }),
  [string]$Distro = "Ubuntu-24.04",
  [string]$InstallPath = "~/robotica/Sim2Real-MJX-JAX-sobre-Linux-WSL",
  [ValidateSet("compatible", "auto", "nvidia", "amd", "intel", "cpu")]
  [string]$Accelerator = "compatible"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step {
  param([string]$Message)
  Write-Host ""
  Write-Host "==> $Message" -ForegroundColor Cyan
}

function Get-WslExecutable {
  $command = Get-Command wsl.exe -ErrorAction SilentlyContinue
  if ($command) { return $command.Source }
  $systemWsl = Join-Path $env:SystemRoot "System32\wsl.exe"
  if (Test-Path -LiteralPath $systemWsl -PathType Leaf) { return $systemWsl }
  return $null
}

function Test-IsAdministrator {
  $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
  $principal = [Security.Principal.WindowsPrincipal]::new($identity)
  return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Invoke-ElevatedAndWait {
  param(
    [Parameter(Mandatory = $true)][string]$Executable,
    [Parameter(Mandatory = $true)][string[]]$ArgumentList,
    [Parameter(Mandatory = $true)][string]$Description,
    [int[]]$AllowedExitCodes = @(0)
  )
  Write-Host "Windows mostrara una confirmacion de administrador para $Description." -ForegroundColor Yellow
  try {
    $process = Start-Process -FilePath $Executable -ArgumentList $ArgumentList -Verb RunAs -Wait -PassThru
  }
  catch {
    throw "No se concedieron permisos de administrador para $Description. $($_.Exception.Message)"
  }
  if ($process.ExitCode -notin $AllowedExitCodes) {
    throw "$Description termino con codigo $($process.ExitCode)."
  }
}

function Ensure-WslCommand {
  $wsl = Get-WslExecutable
  if ($wsl) { return $wsl }

  $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
  if (-not $winget) {
    throw "Este Windows no incluye todavia WSL ni winget. Actualiza Windows 10/11 y vuelve a ejecutar el mismo bloque."
  }

  Write-Step "Instalando el componente actual de WSL"
  $arguments = @(
    "install", "--id", "Microsoft.WSL", "--exact", "--source", "winget",
    "--accept-package-agreements", "--accept-source-agreements"
  )
  if (Test-IsAdministrator) {
    & $winget.Source @arguments
    if ($LASTEXITCODE -notin @(0, 1641, 3010)) {
      throw "winget no pudo instalar Microsoft.WSL (codigo $LASTEXITCODE)."
    }
  }
  else {
    Invoke-ElevatedAndWait -Executable $winget.Source -ArgumentList $arguments `
      -Description "instalar WSL" -AllowedExitCodes @(0, 1641, 3010)
  }

  $wsl = Get-WslExecutable
  if (-not $wsl) {
    Write-Host "WSL se ha instalado, pero Windows necesita reiniciarse antes de continuar." -ForegroundColor Yellow
    Write-Host "Reinicia y vuelve a pegar exactamente el mismo bloque del README." -ForegroundColor Yellow
    return $null
  }
  return $wsl
}

function Invoke-Wsl {
  param([string]$Command)
  & $script:WslExecutable -d $Distro -- bash -lc $Command
  if ($LASTEXITCODE -ne 0) {
    throw "El comando WSL termino con codigo $LASTEXITCODE."
  }
}

function Invoke-WslScript {
  param(
    [Parameter(Mandatory = $true)][string]$Script,
    [string[]]$ArgumentList = @(),
    [switch]$CaptureOutput,
    [switch]$AsRoot
  )

  $capturedLines = [System.Collections.Generic.List[string]]::new()
  # El propio instalador se descarga con CRLF en Windows. El script Bash se
  # guarda en un archivo temporal UTF-8 sin BOM y con LF para que ni la tuberia
  # de Windows PowerShell ni su codificacion puedan alterar el contenido.
  $normalizedScript = $Script -replace "`r`n", "`n"
  $temporaryScript = Join-Path ([IO.Path]::GetTempPath()) (
    "sim2real-wsl-" + [guid]::NewGuid().ToString("N") + ".sh"
  )
  $utf8WithoutBom = [Text.UTF8Encoding]::new($false)
  [IO.File]::WriteAllText($temporaryScript, $normalizedScript, $utf8WithoutBom)
  $previousTemporaryPath = [Environment]::GetEnvironmentVariable(
    "SIM2REAL_TEMP_SCRIPT_WINDOWS", "Process"
  )
  $previousWslEnv = [Environment]::GetEnvironmentVariable("WSLENV", "Process")
  $env:SIM2REAL_TEMP_SCRIPT_WINDOWS = $temporaryScript
  $wslEnvEntries = @(
    @($previousWslEnv -split ':') |
      Where-Object { $_ -and $_ -notmatch '^SIM2REAL_TEMP_SCRIPT_WINDOWS(?:/.*)?$' }
  )
  $env:WSLENV = (@($wslEnvEntries) + "SIM2REAL_TEMP_SCRIPT_WINDOWS/p") -join ':'
  try {
    $wslArguments = @("-d", $Distro)
    if ($AsRoot) { $wslArguments += @("-u", "root") }
    $wslArguments += "--"
    $linuxPathLines = @(
      & $script:WslExecutable @wslArguments printenv SIM2REAL_TEMP_SCRIPT_WINDOWS
    )
    $pathStatus = $LASTEXITCODE
    if ($pathStatus -ne 0 -or $linuxPathLines.Count -ne 1) {
      throw "WSL no pudo recibir la ruta del script temporal (codigo $pathStatus)."
    }
    $linuxScript = ([string]$linuxPathLines[0]).Trim()
    if (-not $linuxScript.StartsWith("/")) {
      throw "WSL devolvio una ruta temporal no valida: $linuxScript"
    }

    & $script:WslExecutable @wslArguments bash $linuxScript @ArgumentList |
      ForEach-Object {
        $line = [string]$_
        Write-Host $line
        if ($CaptureOutput) {
          $capturedLines.Add($line)
        }
      }
    $wslStatus = $LASTEXITCODE
    if ($wslStatus -ne 0) {
      throw "El script WSL termino con codigo $wslStatus."
    }
    if ($CaptureOutput) {
      return $capturedLines.ToArray()
    }
  }
  finally {
    if ($null -eq $previousTemporaryPath) {
      Remove-Item Env:SIM2REAL_TEMP_SCRIPT_WINDOWS -ErrorAction SilentlyContinue
    }
    else {
      $env:SIM2REAL_TEMP_SCRIPT_WINDOWS = $previousTemporaryPath
    }
    if ($null -eq $previousWslEnv) {
      Remove-Item Env:WSLENV -ErrorAction SilentlyContinue
    }
    else {
      $env:WSLENV = $previousWslEnv
    }
    Remove-Item -LiteralPath $temporaryScript -Force -ErrorAction SilentlyContinue
  }
}

function Start-WslDistributionInstall {
  param([int]$PreviousStatus = 0)
  Write-Step "Instalando WSL2 y $Distro"
  & $script:WslExecutable --install -d $Distro
  $installStatus = $LASTEXITCODE
  if ($installStatus -ne 0) {
    if ($PreviousStatus -ne 0) {
      throw "No se pudo consultar WSL (codigo $PreviousStatus) ni iniciar la instalacion de $Distro (codigo $installStatus). Comprueba que PowerShell esta abierto como administrador y que Windows Update esta al dia."
    }
    throw "Windows no pudo instalar $Distro (codigo $installStatus). Abre PowerShell como administrador y comprueba Windows Update."
  }
  Write-Host ""
  Write-Host "Windows ha iniciado la instalacion. Reinicia si lo solicita, abre $Distro una vez para crear tu usuario y repite este mismo script." -ForegroundColor Yellow
}

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
  throw "Este bootstrap se ejecuta desde PowerShell en Windows. En Ubuntu usa ./scripts/install.sh."
}
trap {
  Write-Host ""
  Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
  exit 1
}
if ([Environment]::OSVersion.Version.Build -lt 19041) {
  throw "WSL2 necesita Windows 10 2004 (build 19041) o una version posterior. Actualiza Windows antes de instalar."
}
$architecture = [Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
if ($architecture -ne "X64") {
  throw "Esta version contiene un entorno reproducible x86_64; la arquitectura $architecture no esta validada."
}
if ($Distro -notmatch '^[A-Za-z0-9._-]+$') {
  throw "Nombre de distro no valido: $Distro"
}
if ($RepoUrl -match "[`r`n`0]" -or $InstallPath -match "[`r`n`0]") {
  throw "RepoUrl e InstallPath no pueden contener saltos de linea ni caracteres nulos."
}

$script:WslExecutable = Ensure-WslCommand
if (-not $script:WslExecutable) { exit 0 }

Write-Step "Comprobando WSL2 y $Distro"
$rawDistros = @(& $script:WslExecutable --list --quiet)
$listStatus = $LASTEXITCODE
if ($listStatus -ne 0) {
  Start-WslDistributionInstall -PreviousStatus $listStatus
  exit 0
}
$distros = $rawDistros -replace "`0", "" | ForEach-Object { $_.Trim() } | Where-Object { $_ }
if ($distros -notcontains $Distro) {
  Start-WslDistributionInstall
  exit 0
}

$wslVersionLines = @(& $script:WslExecutable --list --verbose) -replace "`0", ""
if ($LASTEXITCODE -ne 0) {
  throw "No se pudo consultar la version de $Distro."
}
$distroPattern = '^\s*\*?\s*' + [regex]::Escape($Distro) + '\s+.*\s([12])\s*$'
$matchingLine = $wslVersionLines | Where-Object { $_ -match $distroPattern } | Select-Object -First 1
if (-not $matchingLine) {
  throw "No se pudo determinar si $Distro utiliza WSL1 o WSL2. Ejecuta 'wsl --list --verbose' y revisa la columna VERSION."
}
if ($matchingLine -notmatch "\s2\s*$") {
  Write-Step "Convirtiendo $Distro a WSL2"
  & $script:WslExecutable --set-version $Distro 2
  if ($LASTEXITCODE -ne 0) { throw "No se pudo convertir $Distro a WSL2." }
}

Write-Step "Inicializando Ubuntu"
Invoke-Wsl "true"
$userIdLines = @(& $script:WslExecutable -d $Distro -- id -u)
$userIdStatus = $LASTEXITCODE
$userId = if ($userIdLines.Count) { ([string]$userIdLines[-1]).Trim() } else { "" }
if ($userIdStatus -ne 0 -or $userId -notmatch '^[0-9]+$') {
  throw "No se pudo comprobar el usuario de Ubuntu. Abre $Distro una vez, crea el usuario Linux y repite este mismo bloque."
}
if ($userId -eq "0") {
  throw "Ubuntu todavia utiliza root como usuario predeterminado. Abre $Distro, crea tu usuario Linux normal y repite este mismo bloque."
}

if ($Accelerator -eq "amd") {
  throw "La GPU AMD no esta admitida bajo WSL2. Usa -Accelerator cpu en Windows o instala el perfil AMD desde Ubuntu 24.04 nativo con ROCm compatible."
}
if ($Accelerator -eq "intel") {
  throw "La GPU Intel no esta admitida por esta pila JAX. Usa -Accelerator cpu; el instalador no fingira aceleracion GPU."
}

Write-Step "Preparando las herramientas de Ubuntu"
$systemPackagesScript = @'
set -euo pipefail
packages=(
  ca-certificates curl git build-essential python3 python3-venv python3-pip
  pkg-config pciutils util-linux libgl1 libegl1 libglfw3 libglew2.2
)
missing=()
for package in "${packages[@]}"; do
  if ! dpkg-query -W -f='${Status}' "${package}" 2>/dev/null | grep -qx 'install ok installed'; then
    missing+=("${package}")
  fi
done
if (( ${#missing[@]} == 0 )); then
  echo "Las herramientas base de Ubuntu ya estan instaladas."
  exit 0
fi
echo "Paquetes pendientes: ${missing[*]}"
apt-get \
  -o Acquire::Retries=3 \
  -o Acquire::http::Timeout=30 \
  -o Acquire::https::Timeout=30 \
  -o DPkg::Lock::Timeout=120 \
  update
DEBIAN_FRONTEND=noninteractive apt-get \
  -o Acquire::Retries=3 \
  -o Acquire::http::Timeout=30 \
  -o Acquire::https::Timeout=30 \
  -o DPkg::Lock::Timeout=120 \
  install -y "${missing[@]}"
'@
Invoke-WslScript -Script $systemPackagesScript -AsRoot

Write-Step "Creando la copia Linux independiente en $InstallPath"
$cloneScript = @'
set -euo pipefail
repo_url="$1"
install_path="$2"
case "${install_path}" in
  "~") expanded_path="${HOME}" ;;
  "~/"*) expanded_path="${HOME}/${install_path#\~/}" ;;
  /*) expanded_path="${install_path}" ;;
  *) echo "InstallPath debe ser absoluto o comenzar por ~/: ${install_path}" >&2; exit 1 ;;
esac
case "/${expanded_path#/}/" in
  */../*|*/./*) echo "InstallPath contiene componentes no permitidos: ${expanded_path}" >&2; exit 1 ;;
esac
mkdir -p "$(dirname "${expanded_path}")"
if [[ -d "${expanded_path}/.git" ]]; then
  if [[ ! -f "${expanded_path}/.sim2real-repository" ]] \
      || [[ "$(<"${expanded_path}/.sim2real-repository")" != "sim2real-mjx-jax-linux-wsl-v1" ]] \
      || [[ ! -f "${expanded_path}/pyproject.toml" ]] \
      || ! grep -Eq '^name[[:space:]]*=[[:space:]]*"sim2real-mjx-jax"[[:space:]]*$' "${expanded_path}/pyproject.toml" \
      || [[ ! -f "${expanded_path}/scripts/install.sh" ]]; then
    echo "La ruta contiene un repositorio Git que no es Sim2Real MJX-JAX sobre Linux/WSL: ${expanded_path}" >&2
    echo "No se modifica su remote, rama ni contenido." >&2
    exit 1
  fi
  if [[ -n "$(git -C "${expanded_path}" status --porcelain)" ]]; then
    echo "Hay cambios locales en ${expanded_path}; no se sobrescriben." >&2
    exit 1
  fi
  git -C "${expanded_path}" remote set-url origin "${repo_url}"
  git -C "${expanded_path}" fetch origin main
  git -C "${expanded_path}" checkout main
  git -C "${expanded_path}" pull --ff-only origin main
elif [[ -e "${expanded_path}" ]]; then
  echo "La ruta ya existe y no es un repositorio Git: ${expanded_path}" >&2
  exit 1
else
  git clone --branch main --single-branch "${repo_url}" "${expanded_path}"
fi
chmod +x "${expanded_path}"/scripts/*.sh "${expanded_path}"/scripts/lib/*.sh
printf '__SIM2REAL_REPO_PATH__=%s\n' "${expanded_path}"
'@
$cloneOutput = @(
  Invoke-WslScript -Script $cloneScript -ArgumentList @($RepoUrl, $InstallPath) -CaptureOutput
)
$repoMarkers = @($cloneOutput | Where-Object { $_ -like '__SIM2REAL_REPO_PATH__=*' })
if ($repoMarkers.Count -ne 1) {
  throw "No se pudo determinar de forma segura la ruta del repositorio dentro de WSL."
}
$repoPath = $repoMarkers[0].Substring('__SIM2REAL_REPO_PATH__='.Length)
if (-not $repoPath.StartsWith('/')) {
  throw "La ruta devuelta por WSL no es absoluta: $repoPath"
}

Write-Step "Instalando SIM2REAL ($Accelerator)"
$installArgs = @("--accelerator", $Accelerator, "--skip-system-packages")
$installScript = @'
set -euo pipefail
repo_path="$1"
shift
cd -- "${repo_path}"
exec ./scripts/install.sh "$@"
'@
$allInstallArgs = @($repoPath) + $installArgs
Invoke-WslScript -Script $installScript -ArgumentList $allInstallArgs

Write-Host ""
Write-Host "Instalacion terminada." -ForegroundColor Green
Write-Host "La copia de trabajo vive solo en Linux: $repoPath"
Write-Host "Abrirla desde Windows: \\wsl.localhost\$Distro$($repoPath -replace '/', '\')"
Write-Host "Diagnostico: wsl -d $Distro --cd $repoPath -- ./scripts/doctor.sh"
