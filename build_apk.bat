@echo off
echo === AudioBoost APK Build ===
echo.

echo [1/3] Building APK...
call buildozer android debug
if errorlevel 1 (
    echo BUILD FAILED
    exit /b 1
)

echo.
echo [2/3] Patching AndroidManifest.xml...
for /r .buildozer %%f in (AndroidManifest.xml) do (
    echo Found: %%f
    powershell -Command "(Get-Content '%%f') -replace 'android:name=\"org.audioboost.AudioBoost\"', 'android:name=\"org.audioboost.AudioBoost\"`n            android:foregroundServiceType=\"mediaProjection\"' | Set-Content '%%f'"
    echo Manifest patched.
)

echo.
echo [3/3] Rebuilding with patched manifest...
call buildozer android debug
if errorlevel 1 (
    echo REBUILD FAILED
    exit /b 1
)

echo.
echo === Build complete! APK in bin/ ===
