[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Assert-True {
  param(
    [Parameter(Mandatory = $true)][bool]$Condition,
    [Parameter(Mandatory = $true)][string]$Message
  )
  if (-not $Condition) {
    throw $Message
  }
}

function Count-Text {
  param(
    [Parameter(Mandatory = $true)][string]$Text,
    [Parameter(Mandatory = $true)][string]$Needle
  )
  return ([regex]::Matches($Text, [regex]::Escape($Needle))).Count
}

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$installer = Join-Path $repositoryRoot "scripts\install_windows.ps1"
Assert-True (Test-Path -LiteralPath $installer -PathType Leaf) "No se encuentra install_windows.ps1."

$tokens = $null
$parseErrors = $null
[void][System.Management.Automation.Language.Parser]::ParseFile(
  $installer,
  [ref]$tokens,
  [ref]$parseErrors
)
if ($parseErrors.Count -ne 0) {
  throw ("Errores de sintaxis PowerShell: " + ($parseErrors.Message -join "; "))
}

$source = Get-Content -LiteralPath $installer -Raw
Assert-True ($source -notmatch '\$output\s*=\s*\$Script\s*\|') "Invoke-WslScript vuelve a almacenar toda la salida antes de mostrarla."
Assert-True ($source -match 'ForEach-Object\s*\{[\s\S]*?Write-Host\s+\$line') "Invoke-WslScript no muestra cada linea a medida que llega."
Assert-True ($source -notmatch 'AllowExperimentalAmdWsl') "Sigue publicada la opcion muerta AllowExperimentalAmdWsl."
Assert-True ($source -match '__SIM2REAL_REPO_PATH__=') "Falta el marcador seguro para recuperar la ruta Linux."
Assert-True ($source -match 'function Ensure-WslCommand') "El instalador no puede preparar WSL cuando falta en un Windows nuevo."
Assert-True ($source -match 'Microsoft\.WSL') "El instalador no contiene la ruta automatica para instalar WSL."

foreach ($documentationPath in @(
  (Join-Path $repositoryRoot "README.md"),
  (Join-Path $repositoryRoot "docs\INSTALACION.md")
)) {
  $documentation = Get-Content -LiteralPath $documentationPath -Raw
  $documentedBlock = [regex]::Match(
    $documentation,
    '(?s)```powershell\r?\n(\$sim2realInstaller.*?\r?\n)```'
  )
  Assert-True $documentedBlock.Success "No se encuentra el autoinstalador PowerShell en $documentationPath."
  $blockTokens = $null
  $blockErrors = $null
  [void][System.Management.Automation.Language.Parser]::ParseInput(
    $documentedBlock.Groups[1].Value,
    [ref]$blockTokens,
    [ref]$blockErrors
  )
  if ($blockErrors.Count -ne 0) {
    throw ("El bloque PowerShell de $documentationPath no compila: " + ($blockErrors.Message -join "; "))
  }
}

$testRoot = Join-Path ([IO.Path]::GetTempPath()) ("sim2real-windows-installer-test-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $testRoot | Out-Null
$fakeWsl = Join-Path $testRoot "wsl.exe"
$fakeLog = Join-Path $testRoot "wsl-arguments.log"
$stdoutFile = Join-Path $testRoot "stdout.log"
$stderrFile = Join-Path $testRoot "stderr.log"
$originalPath = $env:PATH
$originalLog = $env:SIM2REAL_FAKE_WSL_LOG
$originalMode = $env:SIM2REAL_FAKE_WSL_MODE

$fakeWslSource = @'
using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;

public static class FakeWsl {
  private static void LogArguments(string[] args) {
    string path = Environment.GetEnvironmentVariable("SIM2REAL_FAKE_WSL_LOG");
    if (String.IsNullOrEmpty(path)) return;
    string encoded = String.Join("\t", args.Select(
      value => Convert.ToBase64String(Encoding.UTF8.GetBytes(value))
    ));
    File.AppendAllText(path, encoded + Environment.NewLine, Encoding.UTF8);
  }

  public static int Main(string[] args) {
    LogArguments(args);
    string mode = Environment.GetEnvironmentVariable("SIM2REAL_FAKE_WSL_MODE") ?? "ok";
    if (args.SequenceEqual(new[] { "--list", "--quiet" })) {
      if (mode == "list-error") return 31;
      Console.WriteLine("Ubuntu-24.04");
      return 0;
    }
    if (args.SequenceEqual(new[] { "--list", "--verbose" })) {
      Console.WriteLine("  NAME            STATE           VERSION");
      Console.WriteLine("* Ubuntu-24.04    Running         2");
      return 0;
    }
    if (args.Contains("bash") && args.Contains("-lc")) {
      Console.WriteLine("WSL preparado");
      return 0;
    }
    if (args.Contains("id") && args.Contains("-u")) {
      Console.WriteLine("1000");
      return 0;
    }
    if (args.Contains("printenv") && args.Last() == "SIM2REAL_TEMP_SCRIPT_WINDOWS") {
      Console.WriteLine("/" + Environment.GetEnvironmentVariable("SIM2REAL_TEMP_SCRIPT_WINDOWS"));
      return 0;
    }
    if (args.Contains("bash")) {
      int bashIndex = Array.IndexOf(args, "bash");
      string scriptPath = bashIndex + 1 < args.Length ? args[bashIndex + 1].TrimStart('/') : "";
      if (bashIndex < 0 || String.IsNullOrEmpty(scriptPath) || !File.Exists(scriptPath)) {
        Console.Error.WriteLine("Falta el script Bash temporal.");
        return 100;
      }
      string script = File.ReadAllText(scriptPath, Encoding.UTF8);
      if (script.Contains("\r")) {
        Console.Error.WriteLine("El script Bash contiene retornos CR de Windows.");
        return 98;
      }
      if (script.Contains("packages=(")) {
        Console.WriteLine("Herramientas base comprobadas");
        return 0;
      }
      if (script.Contains("__SIM2REAL_REPO_PATH__=")) {
        Console.WriteLine("Repositorio comprobado");
        Console.WriteLine("__SIM2REAL_REPO_PATH__=/home/prueba/Sim2Real-MJX-JAX-sobre-Linux-WSL");
        return 0;
      }
      if (script.Contains("exec ./scripts/install.sh")) {
        Console.WriteLine("INSTALACION_EN_DIRECTO_INICIO");
        Console.Out.Flush();
        Thread.Sleep(100);
        Console.WriteLine("INSTALACION_EN_DIRECTO_FIN");
        return 0;
      }
    }
    Console.Error.WriteLine("Invocacion WSL no prevista: " + String.Join(" ", args));
    return 97;
  }
}
'@

try {
  Add-Type -TypeDefinition $fakeWslSource -Language CSharp `
    -OutputAssembly $fakeWsl -OutputType ConsoleApplication
  $env:PATH = "$testRoot;$originalPath"
  $env:SIM2REAL_FAKE_WSL_LOG = $fakeLog
  $env:SIM2REAL_FAKE_WSL_MODE = "ok"

  $installPath = "~/robotica/prueba con espacio"
  $process = Start-Process -FilePath "powershell.exe" -NoNewWindow -PassThru -Wait `
    -RedirectStandardOutput $stdoutFile -RedirectStandardError $stderrFile `
    -ArgumentList @(
      "-NoProfile",
      "-NonInteractive",
      "-ExecutionPolicy", "Bypass",
      "-File", ('"' + $installer + '"'),
      "-Distro", "Ubuntu-24.04",
      "-InstallPath", ('"' + $installPath + '"'),
      "-Accelerator", "cpu"
    )
  $stdout = Get-Content -LiteralPath $stdoutFile -Raw
  $stderr = Get-Content -LiteralPath $stderrFile -Raw
  Assert-True ($process.ExitCode -eq 0) "El recorrido hermetico termino con $($process.ExitCode): $stdout $stderr"
  Assert-True ((Count-Text $stdout "Repositorio comprobado") -eq 1) "La salida del clonado no aparece exactamente una vez."
  Assert-True ((Count-Text $stdout "__SIM2REAL_REPO_PATH__=") -eq 1) "El marcador de ruta aparece duplicado o falta."
  Assert-True ((Count-Text $stdout "INSTALACION_EN_DIRECTO_INICIO") -eq 1) "Falta el inicio de la salida del instalador."
  Assert-True ((Count-Text $stdout "INSTALACION_EN_DIRECTO_FIN") -eq 1) "Falta el final de la salida del instalador."
  Assert-True ($stdout.IndexOf("INSTALACION_EN_DIRECTO_INICIO") -lt $stdout.IndexOf("INSTALACION_EN_DIRECTO_FIN")) "La salida del instalador no conserva su orden."

  $argumentCalls = @(Get-Content -LiteralPath $fakeLog | ForEach-Object {
    @($_ -split "`t" | ForEach-Object {
      [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($_))
    })
  })
  $flattenedArguments = $argumentCalls | ForEach-Object { $_ }
  Assert-True ($flattenedArguments -contains $installPath) "InstallPath no llego a WSL como un argumento literal."
  Assert-True ($flattenedArguments -contains "--accelerator") "Falta --accelerator en la llamada a Linux."
  Assert-True ($flattenedArguments -contains "cpu") "El perfil CPU no llego a Linux."

  $env:SIM2REAL_FAKE_WSL_MODE = "list-error"
  Remove-Item -LiteralPath $stdoutFile, $stderrFile -Force
  $failedProcess = Start-Process -FilePath "powershell.exe" -NoNewWindow -PassThru -Wait `
    -RedirectStandardOutput $stdoutFile -RedirectStandardError $stderrFile `
    -ArgumentList @(
      "-NoProfile",
      "-NonInteractive",
      "-ExecutionPolicy", "Bypass",
      "-File", ('"' + $installer + '"'),
      "-Distro", "Ubuntu-24.04",
      "-Accelerator", "cpu"
    )
  $failedOutput = (Get-Content -LiteralPath $stdoutFile -Raw) + (Get-Content -LiteralPath $stderrFile -Raw)
  Assert-True ($failedProcess.ExitCode -ne 0) "Un fallo de wsl --list se presento como instalacion correcta."
  Assert-True ($failedOutput -match 'No se pudo consultar WSL \(codigo 31\)') "El fallo de wsl --list no tiene un diagnostico claro."
}
finally {
  $env:PATH = $originalPath
  $env:SIM2REAL_FAKE_WSL_LOG = $originalLog
  $env:SIM2REAL_FAKE_WSL_MODE = $originalMode
  if (Test-Path -LiteralPath $testRoot) {
    Remove-Item -LiteralPath $testRoot -Recurse -Force
  }
}

Write-Host "Pruebas del instalador Windows: OK" -ForegroundColor Green
