"""
Buildozer hook: adds foregroundServiceType to the AudioBoost service
in AndroidManifest.xml after p4a generates it.
"""
import os
import subprocess


def hook_buildozer(config):
    """Called by buildozer after build. Patches the manifest."""
    manifest = None
    # Find the generated AndroidManifest.xml
    for root, dirs, files in os.walk('.buildozer'):
        if 'AndroidManifest.xml' in files:
            manifest = os.path.join(root, 'AndroidManifest.xml')
            break

    if manifest is None:
        print("[Hook] AndroidManifest.xml not found, skipping patch")
        return

    with open(manifest, 'r') as f:
        content = f.read()

    # Check if already patched
    if 'foregroundServiceType' in content:
        print("[Hook] Manifest already patched")
        return

    # Add foregroundServiceType="mediaProjection" to the service
    old = 'android:name="org.audioboost.AudioBoost"'
    new = 'android:name="org.audioboost.AudioBoost"\n            android:foregroundServiceType="mediaProjection"'

    if old in content:
        content = content.replace(old, new)
        with open(manifest, 'w') as f:
            f.write(content)
        print("[Hook] Manifest patched: foregroundServiceType=mediaProjection")
    else:
        print("[Hook] Service not found in manifest, trying alternate pattern")
        # Try alternate pattern
        old2 = 'android:name="org.audioboost.AudioBoost" />'
        new2 = 'android:name="org.audioboost.AudioBoost"\n            android:foregroundServiceType="mediaProjection" />'
        if old2 in content:
            content = content.replace(old2, new2)
            with open(manifest, 'w') as f:
                f.write(content)
            print("[Hook] Manifest patched (alternate)")
        else:
            print("[Hook] Could not find service in manifest")
