param(
    [string]$DataFolder = "D:\data",
    [string]$TablePath = "D:\data\tables\spirals_ephys_sessions_new2.csv",
    [int]$IntervalSec = 30,
    [switch]$Watch
)

$ErrorActionPreference = "SilentlyContinue"
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch {}

$ephys = Join-Path $DataFolder "ephys"
$procId = 42604

$stages = @(
    @{ Cell = 2; Folder = "dv_prediction";         Short = "dV prediction" },
    @{ Cell = 3; Folder = "dv_permute";            Short = "dV permute" },
    @{ Cell = 4; Folder = "roi";                   Short = "ROI (skip-if-exists)" },
    @{ Cell = 5; Folder = "spirals_raw";           Short = "raw spiral detection" },
    @{ Cell = 6; Folder = "spirals_raw_fftn";      Short = "spiral grouping" },
    @{ Cell = 7; Folder = "spirals_predict";       Short = "spiral prediction" },
    @{ Cell = 8; Folder = "spirals_predict_permute"; Short = "prediction permute" },
    @{ Cell = 9; Folder = "spirals_compare";       Short = "compare predict" },
    @{ Cell = 10; Folder = "spirals_compare";      Short = "compare permute" },
    @{ Cell = 11; Folder = "flow_var";             Short = "phase-flow index" },
    @{ Cell = 12; Folder = "var_ordered";          Short = "var by neuron" }
)

$sessions = @(Import-Csv -LiteralPath $TablePath)
$nSess = $sessions.Count

function Get-Fname($row) {
    $d = [datetime]::ParseExact($row.date, "M/d/yyyy", [Globalization.CultureInfo]::InvariantCulture)
    "{0}_{1:d4}{2:d2}{3:d2}_{4}" -f $row.MouseID, $d.Year, $d.Month, $d.Day, [int]$row.folder
}

function Bar($frac, $width) {
    if ($frac -lt 0) { $frac = 0 }; if ($frac -gt 1) { $frac = 1 }
    $n = [math]::Round($frac * $width)
    ("{0}" -f [string][char]0x2588) * $n + ("{0}" -f [string][char]0x2591) * ($width - $n)
}

function Show-Snapshot {
    $today = (Get-Date).Date
    $alive = [bool](Get-Process -Id $procId -ErrorAction SilentlyContinue)

    $info = foreach ($s in $stages) {
        $files = @(Get-ChildItem -LiteralPath (Join-Path $ephys $s.Folder) -File)
        $todayFiles = @($files | Where-Object { $_.LastWriteTime -ge $today })
        [PSCustomObject]@{
            Cell = $s.Cell; Folder = $s.Folder; Short = $s.Short
            Total = $files.Count; Today = $todayFiles.Count
            Newest = ($files | Sort-Object LastWriteTime -Descending | Select-Object -First 1).LastWriteTime
        }
    }

    # active stage = last stage (in cell order) with a write today
    $activeIdx = -1
    for ($i = 0; $i -lt $info.Count; $i++) {
        if ($info[$i].Today -gt 0) { $activeIdx = $i }
    }
    if ($activeIdx -lt 0) { $activeIdx = 0 }

    Clear-Host
    Write-Host ("pipeline4_ephys.ipynb  |  runner PID {0} {1}  |  {2}  |  {3} sessions" -f `
        $procId, $(if ($alive) {"RUNNING"} else {"NOT RUNNING"}), (Get-Date), $nSess) -ForegroundColor $(if ($alive) {"Green"} else {"Red"})
    Write-Host ("=" * 78)

    $doneCells = 0
    for ($i = 0; $i -lt $info.Count; $i++) {
        $st = $info[$i]
        $isCompare = $st.Folder -eq "spirals_compare"   # cells 9+10 share it; completion only when a later stage started
        $laterStarted = $false
        for ($j = $i + 1; $j -lt $info.Count; $j++) {
            if ($info[$j].Folder -ne $st.Folder -and $info[$j].Today -gt 0) { $laterStarted = $true }
        }
        $isActive = ($i -eq $activeIdx) -and $alive

        if ($isActive) {
            $state = "RUNNING"
            $color = "Yellow"
            $frac = if ($isCompare) { 0.05 } else { $st.Today / $nSess }
        }
        elseif ($laterStarted -or (-not $isCompare -and $st.Today -ge $nSess)) {
            $state = "done"
            $color = "Green"
            $frac = 1.0; $doneCells++
        }
        elseif ($st.Folder -eq "roi") {
            $state = "skipped (exists)"; $color = "DarkGray"; $frac = 1.0; $doneCells++
        }
        else {
            $state = "pending"; $color = "DarkGray"; $frac = 0.0
        }

        $detail = if ($isCompare) { "{0,3} file(s)" -f $st.Total } else { "{0,2}/{1,-2} today" -f $st.Today, $nSess }
        Write-Host ("cell {0,2} {1,-22} {2} {3,5:P0}  {4,-18} {5}" -f `
            $st.Cell, $st.Short, (Bar $frac 20), $frac, $state, $detail) -ForegroundColor $color
    }

    Write-Host ("=" * 78)
    $overall = $doneCells / $info.Count
    Write-Host ("overall  {0} {1,5:P1}  ({2}/{3} stages finished)" -f (Bar $overall 30), $overall, $doneCells, $info.Count) -ForegroundColor Cyan

    $act = $info[$activeIdx]
    if ($act.Today -gt 0 -and $act.Today -le $nSess -and $alive) {
        $idx = $act.Today  # sessions processed in table order; next one has index = written count
        if ($idx -lt $sessions.Count) {
            Write-Host ("active session {0}/{1}: {2}" -f ($idx + 1), $nSess, (Get-Fname $sessions[$idx])) -ForegroundColor Yellow
        }
        if ($act.Today -ge 2) {
            $ts = @(Get-ChildItem -LiteralPath (Join-Path $ephys $act.Folder) -File |
                Where-Object { $_.LastWriteTime -ge $today } | Sort-Object LastWriteTime |
                Select-Object -ExpandProperty LastWriteTime)
            $span = ($ts[-1] - $ts[0]).TotalMinutes / ($ts.Count - 1)
            $remain = ($nSess - $act.Today) * $span
            Write-Host ("stage rate ~{0:N0} min/session  |  est. remaining this stage: {1:N1} h" -f $span, ($remain / 60)) -ForegroundColor DarkYellow
        } else {
            Write-Host "stage rate: need >= 2 finished sessions to estimate" -ForegroundColor DarkGray
        }
    }
    if (-not $alive) {
        Write-Host "runner is not running - check the launching console for DONE/ERROR." -ForegroundColor Red
    }
}

if ($Watch) {
    while ($true) { Show-Snapshot; Start-Sleep -Seconds $IntervalSec }
} else {
    Show-Snapshot
}
