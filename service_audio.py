"""
AudioBoost Android Service — MediaProjection Audio Capture
Capturează audio din sistem (Spotify, YouTube, etc.) prin AudioPlaybackCapture,
procesează prin DSP (EQ, compressor, ambience, etc.) și redă prin AudioTrack.

Citește parametrii DSP din SharedPreferences (scrise de UI) la fiecare 500ms.
Suportă selectarea dispozitivului de ieșire (difuzor, căști, Bluetooth).
"""
import threading
import json


def run():
    print("[AudioBoost] Service starting...")
    try:
        from jnius import autoclass
        Context = autoclass('android.content.Context')
        PythonService = autoclass('org.kivy.android.PythonService')
        service = autoclass('org.kivy.android.PythonService').mService
        if service is None:
            service = PythonService.mService

        intent = service.getIntent()
        result_code = intent.getIntExtra('resultCode', -1)
        result_data = intent.getParcelableExtra('resultData')

        if result_code == -1 or result_data is None:
            print("[AudioBoost] Service: no MediaProjection data")
            service.stopForeground(True)
            service.stopSelf()
            return

        # Foreground notification (required for MediaProjection)
        try:
            NotificationChannel = autoclass('android.app.NotificationChannel')
            NotificationManager = autoclass('android.app.NotificationManager')
            nm = service.getSystemService(Context.NOTIFICATION_SERVICE)
            channel = NotificationChannel(
                "audioboost", "AudioBoost",
                NotificationManager.IMPORTANCE_LOW)
            channel.setDescription("Audio processing active")
            nm.createNotificationChannel(channel)
        except Exception:
            pass

        try:
            Builder = autoclass('android.app.Notification$Builder')
            n = Builder(service, "audioboost")
            n.setContentTitle("AudioBoost")
            n.setContentText("Processing audio...")
            n.setSmallIcon(0x01080041)  # android.R.drawable.ic_media_play
            n.setOngoing(True)
            service.startForeground(1, n.build())
        except Exception as e:
            print(f"[AudioBoost] Notification error: {e}")

        # MediaProjection
        MProjectionManager = autoclass('android.media.projection.MediaProjectionManager')
        mp_mgr = service.getSystemService(Context.MEDIA_PROJECTION_SERVICE)
        projection = mp_mgr.getMediaProjection(result_code, result_data)

        if projection is None:
            print("[AudioBoost] Failed to create MediaProjection")
            service.stopForeground(True)
            service.stopSelf()
            return

        # AudioPlaybackCapture — capture system audio (Spotify, YouTube, games)
        AudioPlaybackCaptureConfig = autoclass(
            'android.media.AudioPlaybackCaptureConfiguration')
        AudioAttributes = autoclass('android.media.AudioAttributes')
        config = AudioPlaybackCaptureConfig.Builder(projection) \
            .addMatchingUsage(AudioAttributes.USAGE_MEDIA) \
            .addMatchingUsage(AudioAttributes.USAGE_GAME) \
            .addMatchingUsage(AudioAttributes.USAGE_UNKNOWN) \
            .build()

        sr = 44100
        ch = 2
        AudioFormat = autoclass('android.media.AudioFormat')
        AudioRecord = autoclass('android.media.AudioRecord')
        AudioTrack = autoclass('android.media.AudioTrack')
        AudioManager = autoclass('android.media.AudioManager')

        enc = AudioFormat.ENCODING_PCM_16BIT
        ch_in = AudioFormat.CHANNEL_IN_STEREO
        ch_out = AudioFormat.CHANNEL_OUT_STEREO
        blocksize = 1024

        buf_sz_in = max(blocksize * ch * 2,
                        AudioRecord.getMinBufferSize(sr, ch_in, enc))
        buf_sz_out = max(blocksize * ch * 2,
                         AudioTrack.getMinBufferSize(sr, ch_out, enc))

        rec = AudioRecord.Builder() \
            .setAudioPlaybackCaptureConfig(config) \
            .setAudioFormat(
                AudioFormat.Builder()
                .setEncoding(enc)
                .setSampleRate(sr)
                .setChannelMask(ch_in).build()) \
            .setBufferSizeInBytes(buf_sz_in) \
            .build()

        trk = AudioTrack(AudioManager.STREAM_MUSIC, sr, ch_out, enc,
                         buf_sz_out, AudioTrack.MODE_STREAM)

        rec.startRecording()
        trk.play()
        print(f"[AudioBoost] Audio started: {sr}Hz, {ch}ch, block={blocksize}")

        # ── DSP initialization ──
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from main import RadioDSP, BLOCKSIZE

        dsp = RadioDSP(sr=sr, ch=ch)
        master_vol = 1.0

        # ── Output device selection ──
        def apply_output_device(device_id):
            """Setează dispozitivul de ieșire pe AudioTrack."""
            if device_id < 0:
                return  # Auto/Default
            try:
                GET_DEVICES_OUTPUTS = 2
                am = service.getSystemService(Context.AUDIO_SERVICE)
                devices = am.getDevices(GET_DEVICES_OUTPUTS)
                for i in range(devices.length):
                    d = devices[i]
                    if int(d.getId()) == device_id:
                        trk.setPreferredDevice(d)
                        print(f"[AudioBoost] Output device set: {d.getProductName()}")
                        return
                print(f"[AudioBoost] Device ID {device_id} not found")
            except Exception as e:
                print(f"[AudioBoost] setPreferredDevice error: {e}")

        # ── SharedPreferences polling ──
        sp = service.getSharedPreferences('audioboost_prefs', 0)
        last_params_json = ""

        def poll_params():
            """Citește parametrii DSP din SharedPreferences."""
            nonlocal last_params_json, master_vol
            try:
                params_json = sp.getString('dsp_params', '{}')
                if params_json == last_params_json:
                    return
                last_params_json = params_json
                p = json.loads(params_json)

                # Apply DSP parameters
                master_vol = float(p.get('master', 1.0))
                dsp.update(
                    float(p.get('in_db', 0)),
                    float(p.get('bd', 4)),
                    float(p.get('td', 3)),
                    float(p.get('pd', 3)),
                    float(p.get('ex', 0.3)),
                    float(p.get('thr', -22)),
                    float(p.get('rat', 5)),
                    float(p.get('mkup', 7)),
                    float(p.get('pmix', 0.4)),
                    float(p.get('sw', 1.5)),
                    float(p.get('haas', 12)),
                    float(p.get('od', 1)),
                    ds=float(p.get('ds', 4)),
                    up=float(p.get('up', 0.3)),
                    amb_wet=float(p.get('amb_wet', 0)),
                    amb_room=float(p.get('amb_room', 0.5)),
                    amb_damp=float(p.get('amb_damp', 0.45)),
                    amb_pre=float(p.get('amb_pre', 15)),
                    sur_str=float(p.get('sur_str', 0)),
                )

                # Apply PEQ bands
                peq = p.get('peq', [])
                for i, band in enumerate(peq):
                    if i < 5:
                        dsp.update_peq(
                            i,
                            float(band.get('freq', 1000)),
                            float(band.get('gain_db', 0)),
                            float(band.get('q', 1.4)),
                            bool(band.get('enabled', False)),
                        )

                # Apply output device
                dev_id = int(p.get('output_device_id', -1))
                apply_output_device(dev_id)

                print(f"[AudioBoost] Params updated: master={master_vol:.2f}")
            except Exception as e:
                print(f"[AudioBoost] poll_params error: {e}")

        # ── Projection stop listener ──
        running = True

        try:
            from jnius import PythonJavaClass, java_method

            class ProjCallback(PythonJavaClass):
                __javainterfaces__ = [
                    'android/media/projection/MediaProjection$Callback']

                @java_method('()V')
                def onStop(self):
                    nonlocal running
                    running = False
                    print("[AudioBoost] Projection stopped")

            projection.registerCallback(ProjCallback(), None)
        except Exception as e:
            print(f"[AudioBoost] Callback error: {e}")

        # ── Main loop ──
        import array as arr
        import numpy as np
        import time as _time

        n_samples = blocksize * ch
        err_count = 0
        poll_counter = 0
        POLL_INTERVAL = 25  # Poll every 25 blocks (~500ms at 44100Hz/1024)

        while running:
            try:
                buf_in = arr.array('h', [0] * n_samples)
                rec.read(buf_in, n_samples)
                in_f32 = np.frombuffer(
                    bytes(buf_in), dtype=np.int16).astype(np.float32) / 32768.0
                indata = in_f32.reshape(-1, ch)

                # Process through DSP
                proc = dsp.process(indata.copy())
                proc = proc * master_vol

                out_i16 = (np.clip(proc, -1, 1) * 32767).astype(np.int16)
                out_bytes = out_i16.tobytes()
                trk.write(out_bytes, len(out_bytes))

                # Poll for parameter changes
                poll_counter += 1
                if poll_counter >= POLL_INTERVAL:
                    poll_counter = 0
                    poll_params()

            except Exception as e:
                err_count += 1
                if err_count <= 5:
                    print(f"[AudioBoost] Loop error: {e}")

        # Cleanup
        rec.stop()
        rec.release()
        trk.stop()
        trk.release()
        projection.stop()
        service.stopForeground(True)
        service.stopSelf()
        print("[AudioBoost] Service stopped")

    except Exception as e:
        print(f"[AudioBoost] Service error: {e}")
        import traceback
        traceback.print_exc()
        try:
            service.stopForeground(True)
            service.stopSelf()
        except Exception:
            pass
