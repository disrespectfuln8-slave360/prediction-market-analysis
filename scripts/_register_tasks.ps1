# Run this once, from an elevated PowerShell window (right-click PowerShell -> Run as Administrator,
# or Win+X -> Terminal (Admin)). Registers 3 scheduled tasks so Apex, the sports-prop logger, and the
# weather backfill loop survive a reboot/crash the same way the other bots already do (auto-restart on
# logon, auto-restart up to 999x on crash). Safe to re-run; -Force overwrites a prior registration.

$base = "C:\Users\birds\prediction-market-analysis\prediction-market-analysis\scripts"

Register-ScheduledTask -TaskName 'PredictionMarketBot-ApexWeather' `
    -Xml (Get-Content "$base\_task_apex.xml" -Raw) -Force

Register-ScheduledTask -TaskName 'PredictionMarketMonitor-SportsPropQuotes' `
    -Xml (Get-Content "$base\_task_sports_quotes.xml" -Raw) -Force

Register-ScheduledTask -TaskName 'PredictionMarketMonitor-WeatherBackfillLoop' `
    -Xml (Get-Content "$base\_task_weather_backfill.xml" -Raw) -Force

Write-Host ""
Write-Host "Registered. Current state:"
Get-ScheduledTask | Where-Object { $_.TaskName -match 'ApexWeather|SportsPropQuotes|WeatherBackfillLoop' } |
    Select-Object TaskName, State
