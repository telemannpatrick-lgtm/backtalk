# backtalk: kill any process whose command line matches Pattern, print how many were killed.
# Lives as a real .ps1 file (not an inline -Command string) because passing
# semicolon-chained, brace-heavy PowerShell through bash's argv translation
# to a native Windows exe is fragile and failed silently when tried inline.
#
# -ProcessId -ne $PID is load-bearing, not defensive fluff: this script's
# own invocation command line contains Pattern's literal text (it's right
# there in argv), so the query matches this very process too. Without the
# exclusion, Stop-Process kills its own host mid-run — silent abnormal
# exit, nothing after it ever runs. Confirmed by reproducing it directly.
param([Parameter(Mandatory)][string]$Pattern)
$procs = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like $Pattern -and $_.ProcessId -ne $PID }
foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
# @(...) forces array context -- a single match comes back as a scalar
# CimInstance, not a 1-element array, and .Count on that silently prints
# nothing instead of "1" (confirmed by reproducing it: a real kill against
# exactly one match reported a blank count, not "1").
Write-Output @($procs).Count
