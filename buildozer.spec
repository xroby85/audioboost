[app]
title = AudioBoost v5
package.name = audioboost
package.domain = org.audioboost

source.dir = .
source.include_exts = py,png,jpg,kv,atlas

version = 5.0

# ── Requirements ──────────────────────────────────────────────
# IMPORTANT: sounddevice pe Android necesită rețeta portaudio din p4a.
# numpy și scipy se compilează din sursă (~15min prima dată, cached ulterior).
# ── Requirements ──────────────────────────────────────────────
# ── Requirements ──────────────────────────────────────────────
# ── Requirements ──────────────────────────────────────────────
requirements = python3,kivy==2.3.0,numpy==1.24.3,scipy==1.10.1,pyjnius
# Fallback fără sounddevice (dacă build pică):
# requirements = python3==3.10.14,kivy==2.3.0,numpy,scipy

# ── Display ───────────────────────────────────────────────────
orientation = portrait
fullscreen = 0

# ── Permisiuni ────────────────────────────────────────────────
android.permissions = RECORD_AUDIO,MODIFY_AUDIO_SETTINGS,INTERNET,WAKE_LOCK

# ── Android API ───────────────────────────────────────────────
android.api = 34
android.minapi = 24
android.ndk = 28c
android.sdk = 34
android.accept_sdk_license = True

# ── Arhitecturi (arm64 = telefoane moderne) ───────────────────
android.archs = arm64-v8a

# ── p4a branch cu suport numpy/scipy actualizat ───────────────
p4a.branch = develop

# ── Gradle ────────────────────────────────────────────────────
android.gradle_dependencies = androidx.appcompat:appcompat:1.6.1

[buildozer]
log_level = 2
warn_on_root = 1
