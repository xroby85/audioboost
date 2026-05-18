[app]
title = AudioBoost v5
package.name = audioboost
package.domain = org.audioboost

source.dir = .
source.include_exts = py,png,jpg,kv,atlas

version = 5.0

requirements = python3,cython==3.0.11,kivy==https://github.com/kivy/kivy/archive/refs/tags/2.3.0.zip,numpy==1.24.3,scipy==1.10.1,pyjnius
orientation = portrait
fullscreen = 0

android.permissions = RECORD_AUDIO, MODIFY_AUDIO_SETTINGS
android.api = 33
android.minapi = 24
android.ndk = 28c
android.accept_sdk_license = True
android.archs = arm64-v8a

[buildozer]
log_level = 2
warn_on_root = 1
p4a.branch = develop
