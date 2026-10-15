# 직캠 스테이지 PC 도우미 (Windows)
# Downloads YouTube videos in original quality (up to 4K) with yt-dlp and hands them to the
# 직캠 스테이지 web app over http://localhost:8790. Started by FancamStage-Helper.bat.
# Written for Windows PowerShell 5.1 (built into Windows 10/11).
param([string]$Origin = '')

$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch {}
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$ProgressPreference = 'SilentlyContinue'

$Version = 1
$Port = 8790
$Root = if ($env:FANCAM_HELPER_ROOT) { $env:FANCAM_HELPER_ROOT } else { Join-Path $env:LOCALAPPDATA 'FancamStage' }
$Bin = Join-Path $Root 'bin'
$Downloads = Join-Path $Root 'downloads'
$Exe = if ($env:OS -eq 'Windows_NT') { '.exe' } else { '' }
$YtDlp = Join-Path $Bin "yt-dlp$Exe"

function Say($text, $color = 'Gray') { Write-Host $text -ForegroundColor $color }

# ---------------------------------------------------------------------------
# Tools: yt-dlp (downloader), ffmpeg (joins 4K video + audio), deno (yt-dlp needs it for YouTube)

function Get-Remote($url, $path) { Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $path }

function Install-FromZip($url, $names) {
  $tmp = Join-Path $Root ('tmp-' + [guid]::NewGuid().ToString('N'))
  New-Item -ItemType Directory -Force $tmp | Out-Null
  try {
    $zip = Join-Path $tmp 'a.zip'
    Get-Remote $url $zip
    Expand-Archive -LiteralPath $zip -DestinationPath $tmp -Force
    foreach ($n in $names) {
      $f = Get-ChildItem -LiteralPath $tmp -Recurse -Filter $n | Select-Object -First 1
      if (-not $f) { throw "$n 을(를) 찾지 못했어요." }
      Copy-Item -LiteralPath $f.FullName -Destination (Join-Path $Bin $n) -Force
    }
  } finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
  }
}

function Install-Tools {
  New-Item -ItemType Directory -Force $Bin, $Downloads | Out-Null
  if (-not (Test-Path -LiteralPath $YtDlp)) {
    Say '· yt-dlp 받는 중…'
    Get-Remote 'https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe' $YtDlp
  } else {
    Say '· yt-dlp 업데이트 확인 중…'
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try { & $YtDlp -U 2>&1 | Out-Null } catch {}
    $ErrorActionPreference = $old
  }
  if (-not (Test-Path -LiteralPath (Join-Path $Bin "ffmpeg$Exe"))) {
    Say '· ffmpeg 받는 중… (처음 한 번, 140MB 정도라 몇 분 걸려요)'
    Install-FromZip 'https://github.com/yt-dlp/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip' @('ffmpeg.exe', 'ffprobe.exe')
  }
  if (-not (Test-Path -LiteralPath (Join-Path $Bin "deno$Exe"))) {
    Say '· deno 받는 중… (처음 한 번)'
    Install-FromZip 'https://github.com/denoland/deno/releases/latest/download/deno-x86_64-pc-windows-msvc.zip' @('deno.exe')
  }
  # yt-dlp finds ffmpeg and deno on PATH.
  $env:PATH = $Bin + [IO.Path]::PathSeparator + $env:PATH
}

# Downloads older than a day were never picked up by the app; tidy them away.
function Clear-Old {
  Get-ChildItem -LiteralPath $Downloads -File -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-1) } |
    Remove-Item -Force -ErrorAction SilentlyContinue
}

# ---------------------------------------------------------------------------
# Download jobs

$Jobs = @{}

function Start-Download($url) {
  $id = (Get-Date).ToString('yyyyMMddHHmmssfff')
  $log = Join-Path $Downloads "$id.log"
  $err = Join-Path $Downloads "$id.err"
  $titleFile = Join-Path $Downloads "$id.title.txt"
  $argv = @(
    '--newline', '--no-playlist', '--no-mtime', '--encoding', 'utf-8',
    # Best video up to 4K plus best audio. Prefer VP9 over AV1: browsers decode it faster.
    '-f', 'bv*+ba/b', '-S', 'res:2160,fps,vcodec:vp9,acodec:opus',
    '--merge-output-format', 'mp4',
    '--print-to-file', '%(title)s', $titleFile,
    '-o', (Join-Path $Downloads "$id.%(ext)s"),
    $url
  )
  # Start-Process joins arguments with spaces, so quote each one ourselves.
  $line = ($argv | ForEach-Object { '"' + ($_ -replace '"', '\"') + '"' }) -join ' '
  $p = Start-Process -FilePath $YtDlp -ArgumentList $line -RedirectStandardOutput $log `
    -RedirectStandardError $err -NoNewWindow -PassThru
  $null = $p.Handle # keeps ExitCode readable after the process ends
  $Jobs[$id] = @{ Proc = $p; Log = $log; Err = $err; TitleFile = $titleFile }
  return $id
}

function Read-Lines($path) {
  if (-not (Test-Path -LiteralPath $path)) { return @() }
  try {
    $fs = [IO.File]::Open($path, 'Open', 'Read', 'ReadWrite')
    $sr = New-Object IO.StreamReader($fs, [Text.Encoding]::UTF8)
    $text = $sr.ReadToEnd()
    $sr.Close()
    return @($text -split "`r?`n")
  } catch { return @() }
}

function Get-OutputFile($id) {
  Get-ChildItem -LiteralPath $Downloads -File |
    Where-Object { $_.BaseName -eq $id -and @('.mp4', '.webm', '.mkv') -contains $_.Extension } |
    Select-Object -First 1
}

function Get-JobState($id) {
  $j = $Jobs[$id]
  if (-not $j) { return @{ state = 'missing' } }
  $lines = Read-Lines $j.Log
  $parts = @($lines | Where-Object { $_ -match '^\[download\] Destination:' }).Count
  $pct = 0.0
  for ($i = $lines.Count - 1; $i -ge 0; $i--) {
    if ($lines[$i] -match '^\[download\]\s+([\d.]+)%') { $pct = [double]::Parse($Matches[1], [Globalization.CultureInfo]::InvariantCulture); break }
  }
  $title = ''
  $t = Read-Lines $j.TitleFile | Where-Object { $_ } | Select-Object -First 1
  if ($t) { $title = $t }

  if ($j.Proc.HasExited) {
    $file = Get-OutputFile $id
    if ($j.Proc.ExitCode -eq 0 -and $file) {
      return @{ state = 'done'; title = $title; size = $file.Length; ext = $file.Extension.TrimStart('.') }
    }
    $msg = (Read-Lines $j.Err | Where-Object { $_ -match 'ERROR' } | Select-Object -Last 1)
    if (-not $msg) { $msg = "yt-dlp가 실패했어요 (코드 $($j.Proc.ExitCode))" }
    return @{ state = 'error'; error = [string]$msg; title = $title }
  }
  if (@($lines | Where-Object { $_ -match '^\[Merger\]' }).Count) {
    return @{ state = 'merging'; title = $title }
  }
  return @{ state = 'downloading'; part = [Math]::Max(1, $parts); percent = $pct; title = $title }
}

function Remove-Job($id) {
  $j = $Jobs[$id]
  if (-not $j) { return }
  if (-not $j.Proc.HasExited) { try { $j.Proc.Kill() } catch {} }
  Get-ChildItem -LiteralPath $Downloads -File | Where-Object { $_.Name -like "$id.*" } |
    Remove-Item -Force -ErrorAction SilentlyContinue
  $Jobs.Remove($id)
}

# ---------------------------------------------------------------------------
# Local web server for the app

function Test-Origin($o) {
  if (-not $o) { return $true } # not a browser page (e.g. typed into the address bar)
  if ($Origin -and $o -eq $Origin) { return $true }
  return $o -match '^http://(localhost|127\.0\.0\.1)(:\d+)?$'
}

function Send-Json($ctx, $obj, [int]$code = 200) {
  $bytes = [Text.Encoding]::UTF8.GetBytes(($obj | ConvertTo-Json -Compress -Depth 5))
  $ctx.Response.StatusCode = $code
  $ctx.Response.ContentType = 'application/json; charset=utf-8'
  $ctx.Response.ContentLength64 = $bytes.Length
  $ctx.Response.OutputStream.Write($bytes, 0, $bytes.Length)
  $ctx.Response.Close()
}

function Send-File($ctx, $file) {
  $ctx.Response.StatusCode = 200
  $ctx.Response.ContentType = if ($file.Extension -eq '.webm') { 'video/webm' } else { 'video/mp4' }
  $ctx.Response.ContentLength64 = $file.Length
  $fs = [IO.File]::OpenRead($file.FullName)
  try { $fs.CopyTo($ctx.Response.OutputStream, 1MB) } finally { $fs.Close(); $ctx.Response.Close() }
}

function Invoke-Request($ctx) {
  $req = $ctx.Request
  $res = $ctx.Response
  $o = $req.Headers['Origin']
  if (-not (Test-Origin $o)) { Send-Json $ctx @{ error = '허용되지 않은 사이트예요.' } 403; return }
  if ($o) {
    $res.AddHeader('Access-Control-Allow-Origin', $o)
    $res.AddHeader('Vary', 'Origin')
    $res.AddHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    $res.AddHeader('Access-Control-Allow-Headers', 'Content-Type')
    $res.AddHeader('Access-Control-Allow-Private-Network', 'true')
  }
  if ($req.HttpMethod -eq 'OPTIONS') { $res.StatusCode = 204; $res.Close(); return }

  $path = $req.Url.AbsolutePath
  $id = $req.QueryString['id']
  if ($path -eq '/status') {
    Send-Json $ctx @{ ok = $true; version = $Version }
  } elseif ($path -eq '/download' -and $req.HttpMethod -eq 'POST') {
    $body = (New-Object IO.StreamReader($req.InputStream, [Text.Encoding]::UTF8)).ReadToEnd() | ConvertFrom-Json
    $url = [string]$body.url
    if ($url -notmatch '^https://(www\.|m\.|music\.)?(youtube\.com|youtu\.be)/') {
      Send-Json $ctx @{ error = '유튜브 링크만 받을 수 있어요.' } 400; return
    }
    $newId = Start-Download $url
    Say "▶ 받기 시작: $url" 'Cyan'
    Send-Json $ctx @{ id = $newId }
  } elseif ($path -eq '/job') {
    Send-Json $ctx (Get-JobState $id)
  } elseif ($path -eq '/file') {
    $file = if ($Jobs[$id]) { Get-OutputFile $id } else { $null }
    if (-not $file) { Send-Json $ctx @{ error = '파일이 없어요.' } 404; return }
    Send-File $ctx $file
  } elseif ($path -eq '/done') {
    Remove-Job $id
    Say '✓ 앱으로 넘겼어요.' 'Green'
    Send-Json $ctx @{ ok = $true }
  } else {
    Send-Json $ctx @{ error = 'not found' } 404
  }
}

# ---------------------------------------------------------------------------

Say ''
Say '  직캠 스테이지 PC 도우미' 'Magenta'
Say ''
try {
  Install-Tools
  Clear-Old
} catch {
  Say "준비하다가 문제가 생겼어요: $($_.Exception.Message)" 'Red'
  Say '인터넷 연결을 확인하고 다시 실행해 주세요.' 'Red'
  return
}

$listener = New-Object Net.HttpListener
$listener.Prefixes.Add("http://localhost:$Port/")
try {
  $listener.Start()
} catch {
  Say '도우미가 이미 켜져 있는 것 같아요. 다른 도우미 창을 확인해 주세요.' 'Yellow'
  return
}

Say ''
Say '  준비 완료! 이 창을 켜 둔 채로 앱에서 유튜브 링크를 넣으세요.' 'Green'
Say '  끄려면 이 창을 닫으면 돼요.' 'Green'
Say ''

while ($listener.IsListening) {
  $ctx = $listener.GetContext()
  try {
    Invoke-Request $ctx
  } catch {
    Say "요청 처리 중 오류: $($_.Exception.Message)" 'Yellow'
    try { $ctx.Response.StatusCode = 500; $ctx.Response.Close() } catch {}
  }
}
