#!/usr/bin/env python3
"""
AudioBoost v5 — Android APK (Kivy)
════════════════════════════════════════════════════════════════
DSP: identic cu versiunea desktop (RadioDSP, TrueAmbience, BinauralSurround3D)
UI:  Kivy — funcționează pe Android, iOS, Windows, Linux

Build APK:
    pip install buildozer
    cd audioboost_android
    buildozer android debug   (prima dată durează ~20 min)
    buildozer android deploy  (instalează pe telefon via USB)
"""

import threading, time, math
import numpy as np

# ── Filtre IIR pure numpy (fără scipy) ────────────────────────
# scipy nu compilează corect pe Android cu Python 3.14

def butter(order, cutoff, btype='low', fs=None):
    """Butterworth IIR — returnează (b, a) biquad. Doar order=2."""
    if fs is None:
        raise ValueError("fs required")
    w0 = 2.0 * math.pi * cutoff / fs
    K = math.tan(w0 / 2.0)
    if order == 1:
        if btype == 'low':
            a0 = 1.0 + K
            b = np.array([K / a0, K / a0])
            a = np.array([1.0, (K - 1.0) / a0])
        elif btype == 'high':
            a0 = 1.0 + K
            b = np.array([1.0 / a0, -1.0 / a0])
            a = np.array([1.0, (K - 1.0) / a0])
        else:
            raise ValueError(f"btype {btype} not supported for order=1")
        return b, a
    elif order == 2:
        sq2 = math.sqrt(2.0)
        if btype == 'low':
            a0 = 1.0 + sq2 * K + K * K
            b = np.array([K*K / a0, 2.0*K*K / a0, K*K / a0])
            a = np.array([1.0, 2.0*(K*K - 1.0)/a0, (1.0 - sq2*K + K*K)/a0])
        elif btype == 'high':
            a0 = 1.0 + sq2 * K + K * K
            b = np.array([1.0 / a0, -2.0 / a0, 1.0 / a0])
            a = np.array([1.0, 2.0*(K*K - 1.0)/a0, (1.0 - sq2*K + K*K)/a0])
        elif btype == 'band':
            # Bandpass peaking la w0 cu Q=1 (lățime ≈ o octavă)
            # Pentru bandpass de bază:
            a0 = 1.0 + K + K * K
            b = np.array([K / a0, 0.0, -K / a0])
            a = np.array([1.0, 2.0*(K*K - 1.0)/a0, (1.0 - K + K*K)/a0])
        else:
            raise ValueError(f"btype {btype} not supported")
        return b, a
    else:
        raise ValueError("Only order=1 and order=2 supported")


def lfilter(b, a, x, zi=None):
    """Filtru IIR direct-form II transposed. x: 1D array. Optimizat pentru viteză."""
    b = np.asarray(b, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    order = max(len(b), len(a)) - 1
    if zi is not None:
        z = np.array(zi, dtype=np.float64)
    else:
        z = np.zeros(order)
    y = np.empty(n, dtype=np.float64)

    if order == 2:
        # Biquad vectorizat — mult mai rapid decât bucla generică
        b0, b1, b2 = b[0], b[1], b[2]
        a1, a2 = a[1], a[2]
        z0, z1 = z[0], z[1]
        for i in range(n):
            xn = x[i]
            yn = b0 * xn + z0
            z0 = b1 * xn - a1 * yn + z1
            z1 = b2 * xn - a2 * yn
            y[i] = yn
        z[0] = z0; z[1] = z1
    elif order == 1:
        b0, b1 = b[0], b[1]
        a1 = a[1]
        z0 = z[0]
        for i in range(n):
            xn = x[i]
            yn = b0 * xn + z0
            z0 = b1 * xn - a1 * yn
            y[i] = yn
        z[0] = z0
    else:
        for i in range(n):
            acc = b[0] * x[i] + z[0]
            for j in range(order - 1):
                z[j] = b[j+1] * x[i] + z[j+1] - a[j+1] * acc
            z[order-1] = b[order] * x[i] - a[order] * acc
            y[i] = acc
    return y, z


def sosfilt_zi(sos):
    """Stare inițială pentru sosfilt (steady-state la intrare 1)."""
    zi = np.zeros((len(sos), 2))
    for s, row in enumerate(sos):
        b0, b1, b2, a0, a1, a2 = row
        dc = (b0 + b1 + b2) / (1.0 + a1 + a2) if (1.0 + a1 + a2) != 0 else 0.0
        zi[s, 0] = dc
        zi[s, 1] = dc
    return zi


def sosfilt(sos, x, zi=None):
    """Cascade de biquad-uri. sos: Nx6, x: 1D, zi: Nx2. Optimizat."""
    y = np.array(x, dtype=np.float64)
    n = len(y)
    n_sec = len(sos)
    if zi is not None:
        z = zi.copy()
    else:
        z = np.zeros((n_sec, 2))
    for s in range(n_sec):
        b0, b1, b2, a0, a1, a2 = sos[s]
        z0, z1 = z[s, 0], z[s, 1]
        # Biquad direct-form II transposed — variabile locale pt viteză
        for i in range(n):
            xn = y[i]
            yn = b0 * xn + z0
            z0 = b1 * xn - a1 * yn + z1
            z1 = b2 * xn - a2 * yn
            y[i] = yn
        z[s, 0] = z0
        z[s, 1] = z1
    return y, z

# ── Detecție platformă ────────────────────────────────────────
try:
    from kivy.utils import platform as _kv_platform
    IS_ANDROID = (_kv_platform == 'android')
except Exception:
    IS_ANDROID = False

# ── Backend audio: sounddevice (PC) sau AudioRecord (Android) ─
SD_OK = False
if not IS_ANDROID:
    try:
        import sounddevice as sd
        SD_OK = True
    except ImportError:
        pass

# ══════════════════════════════════════════════════════════════
#   DSP — identic cu versiunea desktop
# ══════════════════════════════════════════════════════════════

BLOCKSIZE = 1024
CHANNELS  = 2


# ══════════════════════════════════════════════════════════════
#   ANDROID AUDIO — AudioRecord + AudioTrack via pyjnius
# ══════════════════════════════════════════════════════════════

class AndroidAudioStream:
    """
    Înlocuiește sounddevice pe Android.
    Folosește AudioRecord (microfon) + AudioTrack (ieșire)
    cu același callback ca sd.Stream.
    """
    def __init__(self, sr, blocksize, channels, callback):
        self.sr = sr; self.bs = blocksize
        self.ch = channels; self._cb = callback
        self._running = False; self._thread = None
        self._rec_session = 0
        self._trk = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread: self._thread.join(timeout=2.0)

    def close(self): self.stop()

    def _loop(self):
        try:
            from jnius import autoclass
            AudioRecord  = autoclass('android.media.AudioRecord')
            AudioTrack   = autoclass('android.media.AudioTrack')
            AudioFormat  = autoclass('android.media.AudioFormat')
            AudioManager = autoclass('android.media.AudioManager')

            # Encoding: PCM 16-bit
            enc   = AudioFormat.ENCODING_PCM_16BIT
            ch_in = AudioFormat.CHANNEL_IN_STEREO if self.ch==2 else AudioFormat.CHANNEL_IN_MONO
            ch_out= AudioFormat.CHANNEL_OUT_STEREO if self.ch==2 else AudioFormat.CHANNEL_OUT_MONO
            src   = 1   # MIC

            buf_sz_in  = max(self.bs * self.ch * 2,
                             AudioRecord.getMinBufferSize(self.sr, ch_in, enc))
            buf_sz_out = max(self.bs * self.ch * 2,
                             AudioTrack.getMinBufferSize(self.sr, ch_out, enc))

            rec = AudioRecord(src, self.sr, ch_in, enc, buf_sz_in)
            trk = AudioTrack(AudioManager.STREAM_MUSIC, self.sr,
                             ch_out, enc, buf_sz_out,
                             AudioTrack.MODE_STREAM)
            rec.startRecording(); trk.play()

            # Store session IDs for effects init
            try:
                self._rec_session = int(rec.getAudioSessionId())
            except Exception:
                self._rec_session = 0
            try:
                self._trk = trk
            except Exception:
                pass

            print(f"[AndroidAudio] Pornit: {self.sr}Hz, {self.ch}ch, block={self.bs}, session={self._rec_session}")

            import array as arr
            n_samples = self.bs * self.ch
            err_count = 0

            while self._running:
                buf_in = arr.array('h', [0] * n_samples)
                rec.read(buf_in, n_samples)
                # Convertim la float32 [-1, 1]
                in_f32 = np.frombuffer(bytes(buf_in), dtype=np.int16).astype(np.float32) / 32768.0
                indata = in_f32.reshape(-1, self.ch)
                outdata = np.zeros_like(indata)
                try:
                    self._cb(indata, outdata, self.bs, None, None)
                except Exception as cb_err:
                    err_count += 1
                    if err_count <= 5:
                        print(f"[AndroidAudio] CB eroare: {cb_err}")
                # Convertim la int16 și scriem BLOCKING (fără pierderi)
                out_i16 = (np.clip(outdata, -1, 1) * 32767).astype(np.int16)
                out_bytes = out_i16.tobytes()
                trk.write(out_bytes, len(out_bytes))

            rec.stop(); rec.release()
            trk.stop(); trk.release()
        except Exception as e:
            print(f"[AndroidAudio] Eroare: {e}")


# ══════════════════════════════════════════════════════════════
#   ANDROID AUDIO EFFECTS — Equalizer + BassBoost nativ
#   Procesează OUTPUT-ul audio al sistemului (nu microfonul)
# ══════════════════════════════════════════════════════════════

class AndroidAudioEffects:
    """
    Folosește android.media.audiofx.Equalizer + DynamicsProcessing
    pentru a procesa audio output la nivel de sistem.

    Equalizer: 5 benzi parametrice (frecvențe fixe, gain -15..+15 dB)
    DynamicsProcessing: EQ parametric cu Q controlabil (Android 9+)
    BassBoost: boost bass nativ
    """
    def __init__(self, session_id=0):
        self._session = session_id
        self._eq = None
        self._dp = None
        self._bass = None
        self._num_bands = 0
        self._band_freqs = []
        self._initialized = False
        self._dp_available = False

    def init(self):
        """Inițializează efectele audio. Fiecare efect are try-except separat."""
        try:
            from jnius import autoclass
        except Exception as e:
            print(f"[AudioFX] jnius indisponibil: {e}")
            return False

        any_ok = False

        # ── Equalizer (disponibil de la API 9) ──
        try:
            Equalizer = autoclass('android.media.audiofx.Equalizer')
            self._eq = Equalizer(0, self._session)
            self._eq.setEnabled(True)
            self._num_bands = int(self._eq.getNumberOfBands())
            self._band_freqs = []
            for i in range(self._num_bands):
                freq_millihz = self._eq.getCenterFreq(i)
                self._band_freqs.append(freq_millihz / 1000.0)
            print(f"[AudioFX] Equalizer: {self._num_bands} benzi, freq={self._band_freqs}")
            any_ok = True
        except Exception as e:
            print(f"[AudioFX] Equalizer indisponibil: {e}")

        # ── BassBoost (API 9+) ──
        try:
            BassBoost = autoclass('android.media.audiofx.BassBoost')
            self._bass = BassBoost(0, self._session)
            self._bass.setEnabled(True)
            print("[AudioFX] BassBoost: activ")
            any_ok = True
        except Exception as e:
            print(f"[AudioFX] BassBoost indisponibil: {e}")

        # ── DynamicsProcessing (API 28+ / Android 9) — EQ cu Q controlabil ──
        try:
            DynamicsProcessing = autoclass('android.media.audiofx.DynamicsProcessing')
            DynamicsProcessingConfig = autoclass('android.media.audiofx.DynamicsProcessing$Config')
            Eq = autoclass('android.media.audiofx.DynamicsProcessing$Eq')
            EqBand = autoclass('android.media.audiofx.DynamicsProcessing$EqBand')
            bands = []
            default_freqs = [80.0, 250.0, 1000.0, 4000.0, 12000.0]
            for f in default_freqs:
                band = EqBand(True, 1.4, f, 0.0)
                bands.append(band)
            eq_stage = Eq(True, 5, bands)
            config = DynamicsProcessingConfig(True, eq_stage, None, None, None)
            self._dp = DynamicsProcessing(0, self._session, config)
            self._dp.setEnabled(True)
            self._dp_available = True
            print("[AudioFX] DynamicsProcessing: activ (EQ parametric cu Q)")
            any_ok = True
        except Exception as e:
            print(f"[AudioFX] DynamicsProcessing indisponibil: {e}")
            self._dp_available = False

        self._initialized = any_ok
        return any_ok

    def set_band_gain(self, band_idx, gain_db):
        """Setează gain-ul pentru o bandă EQ (-1500..+1500 millibel)."""
        if not self._initialized or self._eq is None:
            return
        if band_idx < 0 or band_idx >= self._num_bands:
            return
        try:
            mb = int(gain_db * 100)  # dB → millibel
            # Clamp la limitele efectului
            lo = int(self._eq.getBandLevelRange()[0])
            hi = int(self._eq.getBandLevelRange()[1])
            mb = max(lo, min(hi, mb))
            self._eq.setBandLevel(band_idx, mb)
        except Exception as e:
            print(f"[AudioFX] set_band_gain eroare: {e}")

    def set_dp_band(self, band_idx, freq, gain_db, q):
        """Setează o bandă DynamicsProcessing (EQ parametric cu Q)."""
        if not self._dp_available or self._dp is None:
            return
        if band_idx < 0 or band_idx >= 5:
            return
        try:
            from jnius import autoclass
            EqBand = autoclass('android.media.audiofx.DynamicsProcessing$EqBand')
            band = EqBand(True, float(q), float(freq), float(gain_db))
            # setEqBand(stage, bandIndex, band)
            # stage: 0=pre-processing, 1=post-processing
            self._dp.setEqBand(1, band_idx, band)
        except Exception as e:
            print(f"[AudioFX] set_dp_band eroare: {e}")

    def set_bass_boost(self, strength_0_1000):
        """Setează BassBoost (0..1000 millibel)."""
        if not self._initialized or self._bass is None:
            return
        try:
            settings = self._bass.getProperties()
            # strength: 0..1000
            s = max(0, min(1000, int(strength_0_1000)))
            # Creăm Settings cu noua valoare
            BassBoostSettings = autoclass('android.media.audiofx.BassBoost$Settings')
            new_settings = BassBoostSettings(f"strength={s}")
            self._bass.setProperties(new_settings)
        except Exception as e:
            print(f"[AudioFX] bass_boost eroare: {e}")

    def get_band_freqs(self):
        """Returnează frecvențele centrale ale bandelor EQ (Hz)."""
        return list(self._band_freqs)

    def get_num_bands(self):
        return self._num_bands

    def has_dynamics_processing(self):
        return self._dp_available

    def release(self):
        """Eliberează resursele."""
        try:
            if self._eq: self._eq.release()
        except: pass
        try:
            if self._bass: self._bass.release()
        except: pass
        try:
            if self._dp: self._dp.release()
        except: pass
        self._initialized = False

def _mc(sos, x, zi):
    ch = x.shape[1]; y = np.empty_like(x); zi_new = np.empty_like(zi)
    for c in range(ch):
        y[:, c], zi_new[:, c, :] = sosfilt(sos, x[:, c], zi=zi[:, c, :])
    return y, zi_new


class BinauralSurround3D:
    """
    3D Surround binaural calitate ridicată pentru căști.
    Straturi:
      1. Cross-feed (Bauer): LP 700Hz + ITD 0.25ms — elimină localizarea in-cap
      2. Virtual rear HRTF: side component + head-shadow LP 3kHz + ITD 0.55ms
      3. Pinna elevation: +3dB@1.5kHz, -6dB@9kHz → percepție înălțime
    """
    def __init__(self, sr):
        self.sr = sr
        # 1. Cross-feed LP 700Hz
        b_cf, a_cf = butter(2, 700.0, 'low', fs=sr)
        self.b_cf = b_cf; self.a_cf = a_cf
        zi_sz = max(len(a_cf), len(b_cf)) - 1
        self._cf_zi = [np.zeros(zi_sz), np.zeros(zi_sz)]
        # Cross-feed delay ~0.25ms
        cf_d = max(2, int(sr * 0.00025))
        self._cf_d = cf_d
        sz = cf_d + BLOCKSIZE * 2 + 8
        self._cf_bL = np.zeros(sz); self._cf_pL = 0
        self._cf_bR = np.zeros(sz); self._cf_pR = 0
        self._cf_m  = sz
        # 2. Head-shadow LP 3kHz (rear)
        b_hs, a_hs = butter(2, 3000.0, 'low', fs=sr)
        self.b_hs = b_hs; self.a_hs = a_hs
        zi_hs = max(len(a_hs), len(b_hs)) - 1
        self._hs_zi = [np.zeros(zi_hs), np.zeros(zi_hs)]
        # Rear ITD ~0.55ms
        rd = max(2, int(sr * 0.00055))
        self._rd = rd
        sz2 = rd + BLOCKSIZE * 2 + 8
        self._rb_L = np.zeros(sz2); self._rp_L = 0
        self._rb_R = np.zeros(sz2); self._rp_R = 0
        self._rm   = sz2
        # 3. Pinna: peaking +3dB@1500Hz Q1.5 și notch -6dB@9000Hz Q2
        def _peak(f0, dBg, Q):
            A=10**(dBg/40); w0=2*math.pi*f0/sr
            sn=math.sin(w0); cs=math.cos(w0); alpha=sn/(2*Q)
            b0=1+alpha*A; b1=-2*cs; b2=1-alpha*A
            a0=1+alpha/A; a1=-2*cs; a2=1-alpha/A
            return np.array([b0/a0,b1/a0,b2/a0]), np.array([1,a1/a0,a2/a0])
        self.b_p1,self.a_p1 = _peak(1500, +3.0, 1.5)
        self.b_p2,self.a_p2 = _peak(9000, -6.0, 2.0)
        zi_p = max(len(self.a_p1), len(self.b_p1)) - 1
        self._p1_zi = [np.zeros(zi_p), np.zeros(zi_p)]
        self._p2_zi = [np.zeros(zi_p), np.zeros(zi_p)]

    def _delay(self, x, buf, pos, d, m):
        n = len(x)
        # Read first (corect și pentru d < BLOCKSIZE)
        rs = (pos - d) % m; re = rs + n
        if re <= m: out = buf[rs:re].copy()
        else: out = np.concatenate([buf[rs:], buf[:re-m]])
        # Then write
        we = pos + n
        if we <= m: buf[pos:we] = x
        else: s=m-pos; buf[pos:]=x[:s]; buf[:we-m]=x[s:]
        return out, we % m

    def process(self, x, strength):
        if strength < 0.01 or x.shape[1] < 2:
            return x
        s = min(1.0, strength)
        L = x[:,0].copy(); R = x[:,1].copy()

        # 1. Cross-feed
        Lf, self._cf_zi[0] = lfilter(self.b_cf, self.a_cf, L, zi=self._cf_zi[0])
        Rf, self._cf_zi[1] = lfilter(self.b_cf, self.a_cf, R, zi=self._cf_zi[1])
        Lfd, self._cf_pL = self._delay(Lf, self._cf_bL, self._cf_pL, self._cf_d, self._cf_m)
        Rfd, self._cf_pR = self._delay(Rf, self._cf_bR, self._cf_pR, self._cf_d, self._cf_m)
        cf = 0.22 * s
        L_cf = L + Rfd * cf
        R_cf = R + Lfd * cf

        # 2. Virtual rear
        side = (L - R) * 0.5
        sL,  self._hs_zi[0] = lfilter(self.b_hs, self.a_hs,  side, zi=self._hs_zi[0])
        sR,  self._hs_zi[1] = lfilter(self.b_hs, self.a_hs, -side, zi=self._hs_zi[1])
        sLd, self._rp_L = self._delay(sL, self._rb_L, self._rp_L, self._rd, self._rm)
        sRd, self._rp_R = self._delay(sR, self._rb_R, self._rp_R, self._rd, self._rm)
        rear = 0.28 * s
        L_out = L_cf - sLd * rear
        R_out = R_cf + sRd * rear

        # 3. Pinna elevation
        pe = 0.35 * s
        Lp, self._p1_zi[0] = lfilter(self.b_p1, self.a_p1, L_out, zi=self._p1_zi[0])
        Lp, self._p2_zi[0] = lfilter(self.b_p2, self.a_p2, Lp,    zi=self._p2_zi[0])
        Rp, self._p1_zi[1] = lfilter(self.b_p1, self.a_p1, R_out, zi=self._p1_zi[1])
        Rp, self._p2_zi[1] = lfilter(self.b_p2, self.a_p2, Rp,    zi=self._p2_zi[1])
        L_out = L_out*(1-pe) + Lp*pe
        R_out = R_out*(1-pe) + Rp*pe

        result = x.copy()
        result[:,0] = x[:,0]*(1-s) + L_out*s
        result[:,1] = x[:,1]*(1-s) + R_out*s
        return result / (1.0 + s * 0.15)


class TrueAmbience:
    _er_L = [2.1, 3.7, 5.3, 7.9, 11.3, 15.1, 19.7, 25.3]
    _er_R = [2.9, 4.3, 6.1, 8.7, 12.1, 16.3, 21.1, 27.7]
    _er_g = [0.40, 0.32, 0.25, 0.19, 0.14, 0.10, 0.07, 0.04]

    def __init__(self, sr):
        self.sr  = sr
        N        = int(sr * 0.080) + 512
        self._N  = N
        ap_ms    = [6.3, 10.7, 18.1, 30.3]
        self._ap_d   = [max(1, int(sr * m / 1000)) for m in ap_ms]
        self._ap_g   = 0.68
        self._ap_L   = [np.zeros(d) for d in self._ap_d]
        self._ap_R   = [np.zeros(d) for d in self._ap_d]
        self._ap_pos = [0] * 4
        self._buf_L  = np.zeros(N)
        self._buf_R  = np.zeros(N)
        self._pos    = 0
        b,  a  = butter(2, 250,  'high', fs=sr)
        b2, a2 = butter(2, 8000, 'low',  fs=sr)
        self._hp_b, self._hp_a = b,  a
        self._lp_b, self._lp_a = b2, a2
        zi_sz = max(len(a), len(b)) - 1
        self._hp_zi  = [np.zeros(zi_sz), np.zeros(zi_sz)]
        zi_sz2 = max(len(a2), len(b2)) - 1
        self._lp_zi  = [np.zeros(zi_sz2), np.zeros(zi_sz2)]

    def _ap(self, x_in, buf, pos, d, g):
        n = len(x_in); out = np.empty(n)
        for i in range(n):
            r = buf[pos % d]; v = x_in[i] - g * r
            out[i] = r + g * v; buf[pos % d] = v
            pos = (pos + 1) % d
        return out, pos

    def process(self, x, amount, size, damp=0.45, pre_ms=15.0):
        if amount < 0.01: return x
        n = x.shape[0]; N = self._N
        # Pre-delay
        pre_samp = max(0, min(int(self.sr * pre_ms / 1000), N - n - 2))
        aL, self._hp_zi[0] = lfilter(self._hp_b, self._hp_a, x[:,0], zi=self._hp_zi[0])
        aR, self._hp_zi[1] = lfilter(self._hp_b, self._hp_a, x[:,1], zi=self._hp_zi[1])
        # Damping HF: damp=0→no LP, damp=1→full LP 6kHz
        lp_fc = max(500.0, 20000.0 * (1.0 - damp * 0.7))
        if damp > 0.05:
            b_d, a_d = butter(1, lp_fc, 'low', fs=self.sr)
            zi_sz = max(len(a_d), len(b_d)) - 1
            if not hasattr(self, '_damp_zi'):
                self._damp_zi = [np.zeros(zi_sz), np.zeros(zi_sz)]
            aL, self._damp_zi[0] = lfilter(b_d, a_d, aL, zi=self._damp_zi[0])
            aR, self._damp_zi[1] = lfilter(b_d, a_d, aR, zi=self._damp_zi[1])
        aL, self._lp_zi[0] = lfilter(self._lp_b, self._lp_a, aL, zi=self._lp_zi[0])
        aR, self._lp_zi[1] = lfilter(self._lp_b, self._lp_a, aR, zi=self._lp_zi[1])
        end = self._pos + n
        if end <= N:
            self._buf_L[self._pos:end] = aL; self._buf_R[self._pos:end] = aR
        else:
            s = N - self._pos
            self._buf_L[self._pos:] = aL[:s]; self._buf_L[:end-N] = aL[s:]
            self._buf_R[self._pos:] = aR[:s]; self._buf_R[:end-N] = aR[s:]

        def rd(buf, ms):
            sc = 0.5 + size * 1.0
            d  = max(1, int(self.sr * ms * sc / 1000)) + pre_samp
            rs = (self._pos - d) % N; re = rs + n
            if re <= N: return buf[rs:re].copy()
            return np.concatenate([buf[rs:], buf[:re-N]])

        eL = np.zeros(n); eR = np.zeros(n)
        for dL, dR, g in zip(self._er_L, self._er_R, self._er_g):
            eL += rd(self._buf_L, dL) * g; eR += rd(self._buf_R, dR) * g

        tL = eL.copy(); tR = eR.copy()
        for k in range(4):
            tL, self._ap_pos[k] = self._ap(tL, self._ap_L[k], self._ap_pos[k], self._ap_d[k], self._ap_g)
            tR, _ = self._ap(tR, self._ap_R[k], self._ap_pos[k], self._ap_d[k], self._ap_g)

        wet = min(amount * 0.70, 0.70)
        out = np.empty_like(x)
        out[:,0] = x[:,0] + (eL*0.8 + tL*0.2) * wet
        out[:,1] = x[:,1] + (eR*0.8 + tR*0.2) * wet
        self._pos = end % N
        return out


class RadioDSP:
    def __init__(self, sr=44100, ch=2):
        self.sr = sr; self.ch = ch
        self.input_db=0.0; self.bass_db=4.0; self.treble_db=3.0
        self.presence_db=3.0; self.exciter=0.30
        self.thresh_db=-22.0; self.ratio=5.0; self.makeup_db=7.0
        self.parallel_mix=0.40; self.stereo_w=1.50; self.haas_ms=12.0
        self.output_db=1.0; self.lim_ceiling=0.96
        self.deesser_db=0.0; self.upscale=0.0
        self.ambience_wet=0.0; self.ambience_room=0.50
        self.ambience_damp=0.45; self.ambience_pre=15.0
        self.surround_str=0.0
        # Parametric EQ — 5 benzi (ca audioboost3)
        self.peq_bands = [
            {'freq': 80.0,    'gain_db': 0.0, 'q': 1.4, 'enabled': False},
            {'freq': 250.0,   'gain_db': 0.0, 'q': 1.4, 'enabled': False},
            {'freq': 1000.0,  'gain_db': 0.0, 'q': 1.4, 'enabled': False},
            {'freq': 4000.0,  'gain_db': 0.0, 'q': 1.4, 'enabled': False},
            {'freq': 12000.0, 'gain_db': 0.0, 'q': 1.4, 'enabled': False},
        ]
        self._lock=threading.Lock(); self._master_lin=1.0
        self._build_filters()

    def _build_filters(self):
        sr=self.sr; ch=self.ch
        def sos(ftype, freq):
            if isinstance(freq, (list, tuple)):
                # Bandpass cu două frecvențe: cascadă de 2 biquad-uri
                b1, a1 = butter(2, freq[0], 'high', fs=sr)
                b2, a2 = butter(2, freq[1], 'low',  fs=sr)
                s = np.array([[b1[0],b1[1],b1[2],1.0,a1[1],a1[2]],
                              [b2[0],b2[1],b2[2],1.0,a2[1],a2[2]]])
            else:
                b, a = butter(2, freq, ftype, fs=sr)
                s = np.array([[b[0],b[1],b[2],1.0,a[1],a[2]]])
            zi = sosfilt_zi(s)
            zi = np.stack([zi]*ch, axis=1)
            return s, zi.copy()
        self.sos_bs, self.zi_bs   = sos('low',  200)
        self.sos_tr, self.zi_tr   = sos('high', 8000)
        self.sos_pr, self.zi_pr   = sos('band', [2000, 6000])
        self.sos_ex, self.zi_ex   = sos('high', 3500)
        self.sos_det,self.zi_det  = sos('high', 80)
        self.sos_ds, self.zi_ds   = sos('band', [5000, min(9000, int(sr*0.40))])
        self._ds_env  = np.full(ch, 1e-4)
        self._ds_atk  = np.exp(-1.0/(sr*0.003))
        self._ds_rel  = np.exp(-1.0/(sr*0.060))
        self._ds_thresh = 10.0**(-30.0/20.0)
        _up_src_lo = min(12000, int(sr*0.27))
        _up_hi_air = min(int(sr*0.43), 18000)
        self.sos_up_src, self.zi_up_src = sos('band', [_up_src_lo, _up_hi_air])
        _up_hp = min(int(sr*0.38), 16000)
        self.sos_up_hp, self.zi_up_hp   = sos('high', _up_hp)
        _up_prs_hi = min(int(sr*0.27), 12000)
        self.sos_up_prs, self.zi_up_prs = sos('band', [6000, _up_prs_hi])
        _up_prs_hp = min(int(sr*0.21), 9000)
        self.sos_up_prs_hp, self.zi_up_prs_hp = sos('high', _up_prs_hp)
        self.env_rms  = np.full(ch, 1e-4)
        self.env_gain = np.ones(ch)
        self._lim_gain = 1.0
        self._lim_atk  = np.exp(-1.0/(sr*0.0005))
        self._lim_rel  = np.exp(-1.0/(sr*0.080))
        self._haas_max = int(sr*0.050)+BLOCKSIZE+4
        self._haas_buf = np.zeros(self._haas_max, dtype=np.float64)
        self._haas_pos = 0
        spb = sr/BLOCKSIZE
        self.alpha_a = np.exp(-1/(spb*5/1000+1e-9))
        self.alpha_r = np.exp(-1/(spb*120/1000+1e-9))
        self._ambience = TrueAmbience(sr)
        self._surround = BinauralSurround3D(sr)
        # PEQ state
        self.zi_peq = [np.zeros((1, ch, 2)) for _ in range(5)]

    @staticmethod
    def _lin(db): return 10.0**(db/20.0)

    @staticmethod
    def _peaking_sos(f0, dB, Q, sr):
        A=10.0**(dB/40.0); f0=max(20.0,min(f0,sr*0.499))
        w0=2.0*math.pi*f0/sr; sn=math.sin(w0); cs=math.cos(w0)
        al=sn/(2.0*max(Q,0.05))
        b0=1+al*A; b1=-2*cs; b2=1-al*A
        a0=1+al/A; a1=-2*cs; a2=1-al/A
        return np.array([[b0/a0,b1/a0,b2/a0,1.0,a1/a0,a2/a0]])

    def _haas_delay(self, sig_r, delay_samp):
        n=len(sig_r); buf=self._haas_buf; pos=self._haas_pos
        end=pos+n
        if end<=self._haas_max: buf[pos:end]=sig_r
        else:
            s=self._haas_max-pos; buf[pos:]=sig_r[:s]; buf[:end-self._haas_max]=sig_r[s:]
        self._haas_pos=end%self._haas_max
        rs=(pos-delay_samp)%self._haas_max; re=rs+n
        if re<=self._haas_max: return buf[rs:re].copy()
        return np.concatenate([buf[rs:],buf[:re-self._haas_max]])

    def _limiter_vec(self, x):
        ceiling=self.lim_ceiling; peak=float(np.max(np.abs(x))); g=self._lim_gain
        n=x.shape[0]; t=np.arange(n,dtype=np.float64)
        if peak*g>ceiling:
            g_target=ceiling/(peak+1e-10)
            env=g_target+(g-g_target)*(self._lim_atk**t)
        else:
            g_tgt=min(1.0,ceiling/(peak+1e-10))
            env=g_tgt+(g-g_tgt)*(self._lim_rel**t)
        self._lim_gain=float(env[-1])
        return x*float(env[0])

    def process(self, audio):
        if audio.ndim==1: audio=audio[:,np.newaxis]
        if audio.shape[1]<self.ch:
            audio=np.tile(audio,(1,self.ch//audio.shape[1]+1))[:,:self.ch]
        elif audio.shape[1]>self.ch:
            audio=audio[:,:self.ch]
        with self._lock:
            in_db=self.input_db; bd=self.bass_db; td=self.treble_db
            pd=self.presence_db; ex=self.exciter
            thr_db=self.thresh_db; rat=self.ratio; mkup_db=self.makeup_db
            pmix=self.parallel_mix; sw=self.stereo_w; haas=self.haas_ms
            od=self.output_db; ds_db=self.deesser_db; up_amt=self.upscale
            amb_wet=self.ambience_wet; amb_room=self.ambience_room
            amb_damp=self.ambience_damp; amb_pre=self.ambience_pre
            sur_str=self.surround_str
            peq_bands = list(self.peq_bands)
        x=audio.astype(np.float64)
        if in_db!=0.0: x*=self._lin(in_db)
        x_bs,self.zi_bs=_mc(self.sos_bs,x,self.zi_bs); x+=x_bs*(self._lin(bd)-1.0)
        x_tr,self.zi_tr=_mc(self.sos_tr,x,self.zi_tr); x+=x_tr*(self._lin(td)-1.0)
        if abs(pd)>0.05:
            x_pr,self.zi_pr=_mc(self.sos_pr,x,self.zi_pr); x+=x_pr*(self._lin(pd)-1.0)
        if ex>0.01:
            x_ex,self.zi_ex=_mc(self.sos_ex,x,self.zi_ex)
            rms_in=np.sqrt(np.mean(x_ex**2)+1e-12)
            sat=(np.tanh(x_ex*2.0)*0.60+np.tanh(x_ex*4.5)*0.28+np.tanh(x_ex*9.0)*0.12)
            rms_sat=np.sqrt(np.mean(sat**2)+1e-12)
            x+=sat*(rms_in/rms_sat)*ex*0.45
        # Parametric EQ — 5 benzi
        for i, band in enumerate(peq_bands):
            if band['enabled'] and abs(band['gain_db']) > 0.05:
                sos_p = self._peaking_sos(band['freq'], band['gain_db'], band['q'], self.sr)
                x, self.zi_peq[i] = _mc(sos_p, x, self.zi_peq[i])
        if ds_db>0.1:
            x_sib,self.zi_ds=_mc(self.sos_ds,x,self.zi_ds)
            rms_sib=np.sqrt(np.mean(x_sib**2,axis=0)+1e-12)
            ds_floor=self._lin(-ds_db)
            for c in range(self.ch):
                r=rms_sib[c]; e=self._ds_env[c]
                a=self._ds_atk if r>e else self._ds_rel
                self._ds_env[c]=a*e+(1.0-a)*r
            gr=np.where(self._ds_env>self._ds_thresh,
                        np.maximum(ds_floor,self._ds_thresh/(self._ds_env+1e-10)),
                        np.ones(self.ch))
            x=x-x_sib*(1.0-gr[np.newaxis,:])
        if up_amt>0.01:
            x_src,self.zi_up_src=_mc(self.sos_up_src,x,self.zi_up_src)
            rms_src=np.sqrt(np.mean(x_src**2)+1e-12)
            x_harm=(np.tanh(x_src*2.5)*0.55+np.tanh(x_src*6.0)*0.30+np.tanh(x_src*12.0)*0.15)
            x_new,self.zi_up_hp=_mc(self.sos_up_hp,x_harm,self.zi_up_hp)
            rms_new=np.sqrt(np.mean(x_new**2)+1e-12)
            x+=x_new*(rms_src/(rms_new+1e-10))*up_amt*0.45
            x_prs,self.zi_up_prs=_mc(self.sos_up_prs,x,self.zi_up_prs)
            rms_prs=np.sqrt(np.mean(x_prs**2)+1e-12)
            x_ph=(np.tanh(x_prs*3.0)*0.60+np.tanh(x_prs*7.0)*0.28+np.tanh(x_prs*14.0)*0.12)
            x_pnew,self.zi_up_prs_hp=_mc(self.sos_up_prs_hp,x_ph,self.zi_up_prs_hp)
            rms_pnew=np.sqrt(np.mean(x_pnew**2)+1e-12)
            x+=x_pnew*(rms_prs/(rms_pnew+1e-10))*up_amt*0.22
        thr=self._lin(thr_db); mkup=self._lin(mkup_db)
        x_det,self.zi_det=_mc(self.sos_det,x,self.zi_det)
        rms_blk=np.sqrt(np.mean(x_det**2,axis=0)+1e-10)
        x_dry=x.copy()
        for c in range(self.ch):
            r=rms_blk[c]; e=self.env_rms[c]
            alpha=self.alpha_a if r>e else self.alpha_r
            e=alpha*e+(1.0-alpha)*r; self.env_rms[c]=e
            gr=(1.0/(1.0+(e/thr-1.0)*(rat-1.0)/rat)) if e>thr else 1.0
            g_t=gr*mkup; g_s=self.env_gain[c]*0.82+g_t*0.18
            self.env_gain[c]=g_s; x[:,c]*=g_s
        if pmix>0.01: x=x*(1.0-pmix)+(x_dry*mkup)*pmix
        if self.ch>=2:
            if haas>0.3:
                ds=max(1,min(int(self.sr*haas/1000.0),self._haas_max-BLOCKSIZE-2))
                x[:,1]=self._haas_delay(x[:,1],ds)
            mid=(x[:,0]+x[:,1])*0.5; side=(x[:,0]-x[:,1])*0.5*sw
            x[:,0]=mid+side; x[:,1]=mid-side
        x=self._ambience.process(x, amb_wet, amb_room, amb_damp, amb_pre)
        x=self._surround.process(x,sur_str)
        if od!=0.0: x*=self._lin(od)
        x=self._limiter_vec(x)
        return x.astype(np.float32)

    def set_master(self,v): self._master_lin=float(v)
    def get_master(self):   return self._master_lin

    def update(self, in_db,bd,td,pd,ex,thr,rat,mkup,pmix,sw,haas,od,
               ds=0.0, up=0.0, amb_wet=0.0, amb_room=0.5,
               amb_damp=0.45, amb_pre=15.0, sur_str=0.0):
        with self._lock:
            self.input_db=in_db; self.bass_db=bd; self.treble_db=td
            self.presence_db=pd; self.exciter=ex; self.thresh_db=thr
            self.ratio=rat; self.makeup_db=mkup; self.parallel_mix=pmix
            self.stereo_w=sw; self.haas_ms=haas; self.output_db=od
            self.deesser_db=ds; self.upscale=up
            self.ambience_wet=amb_wet; self.ambience_room=amb_room
            self.ambience_damp=amb_damp; self.ambience_pre=amb_pre
            self.surround_str=sur_str

    def update_peq(self, idx: int, freq: float, gain_db: float,
                   q: float, enabled: bool):
        with self._lock:
            b = self.peq_bands[idx]
            if abs(b['freq'] - freq) > 1.0:
                self.zi_peq[idx] = np.zeros((1, self.ch, 2))
            b['freq']=freq; b['gain_db']=gain_db; b['q']=q; b['enabled']=enabled


# ══════════════════════════════════════════════════════════════
#   PRESETURI
# ══════════════════════════════════════════════════════════════

PRESETS = {
    "Kiss FM":     dict(in_db=0,bd=5,  td=3.5,pd=4,  ex=0.35,thr=-22,rat=5,mkup=8, pmix=0.40,sw=1.55,haas=12,od=1.0, ds=4.0,up=0.30,amb_wet=0.0,amb_room=0.50,sur_str=0.0),
    "Kiss FM Pro": dict(in_db=0,bd=4.5,td=3,  pd=3.5,ex=0.40,thr=-20,rat=4,mkup=7, pmix=0.35,sw=1.60,haas=10,od=0.8, ds=6.0,up=0.45,amb_wet=0.0,amb_room=0.50,sur_str=0.0),
    "Boom3D":      dict(in_db=0,bd=3,  td=4,  pd=2,  ex=0.45,thr=-18,rat=3,mkup=5, pmix=0.25,sw=1.80,haas=14,od=0.5, ds=5.0,up=0.55,amb_wet=0.40,amb_room=0.55,sur_str=0.6),
    "Rock FM":     dict(in_db=0,bd=3,  td=4,  pd=4,  ex=0.40,thr=-20,rat=4,mkup=6, pmix=0.30,sw=1.60,haas=8, od=0.5, ds=4.0,up=0.20,amb_wet=0.0,amb_room=0.50,sur_str=0.0),
    "Bass Club":   dict(in_db=0,bd=8,  td=1,  pd=1,  ex=0.10,thr=-24,rat=6,mkup=9, pmix=0.45,sw=1.40,haas=0, od=1.5, ds=0.0,up=0.00,amb_wet=0.0,amb_room=0.50,sur_str=0.0),
    "Crisp HiFi":  dict(in_db=0,bd=0,  td=5,  pd=5,  ex=0.50,thr=-16,rat=2,mkup=3, pmix=0.15,sw=1.70,haas=15,od=0.0, ds=5.0,up=0.65,amb_wet=0.20,amb_room=0.45,sur_str=0.0),
    "Flat":        dict(in_db=0,bd=0,  td=0,  pd=0,  ex=0.00,thr=-10,rat=1,mkup=0, pmix=0.00,sw=1.00,haas=0, od=0.0, ds=0.0,up=0.00,amb_wet=0.0,amb_room=0.50,sur_str=0.0),
}

# ── EQ Presets (5 benzi parametrice) — identice cu PC ────────
EQ_PRESETS = {
    "Dance / House": [
        (  60.0,  +4.5, 1.5, True),
        ( 250.0,  -2.5, 2.0, True),
        ( 800.0,  -1.0, 1.5, True),
        (3000.0,  +2.0, 1.4, True),
        (10000.0, +3.5, 1.2, True),
    ],
    "Pop": [
        ( 80.0,  +1.5, 1.2, True),
        ( 250.0, -1.0, 1.4, True),
        (1000.0, +2.5, 1.0, True),
        (3500.0, +1.5, 1.4, True),
        (12000.0, +1.0, 1.0, True),
    ],
    "Gaming / 3D": [
        ( 100.0, +3.0, 1.0, True),
        ( 400.0, -2.0, 1.4, True),
        (1500.0, +1.5, 1.2, True),
        (4500.0, +3.5, 2.5, True),
        (12000.0, +2.0, 1.0, True),
    ],
    "Vocal / Podcast": [
        ( 80.0,  -4.0, 1.0, True),
        ( 200.0, +1.5, 1.2, True),
        (1200.0, +2.0, 1.5, True),
        (5000.0, +1.0, 1.4, True),
        (10000.0,-1.5, 1.0, True),
    ],
    "Rock FM": [
        (  80.0, +3.0, 1.2, True),
        ( 250.0, -1.5, 1.6, True),
        ( 800.0, -1.0, 1.5, True),
        (3500.0, +2.5, 1.2, True),
        (12000.0, +3.0, 1.0, True),
    ],
    "Bass Club": [
        (  60.0, +6.0, 1.2, True),
        ( 150.0, +3.0, 1.5, True),
        ( 500.0, -3.0, 2.0, True),
        (3000.0, +1.0, 1.4, True),
        (10000.0, +2.0, 1.0, True),
    ],
    "Flat (Reset)": [
        ( 80.0,   0.0, 1.4, False),
        ( 250.0,  0.0, 1.4, False),
        (1000.0,  0.0, 1.4, False),
        (4000.0,  0.0, 1.4, False),
        (12000.0, 0.0, 1.4, False),
    ],
}

# ══════════════════════════════════════════════════════════════
#   KIVY UI
# ══════════════════════════════════════════════════════════════

from kivy.app import App as KivyApp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.slider import Slider
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.spinner import Spinner
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle, Line, Ellipse
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.metrics import dp, sp

Window.clearcolor = (0.031, 0.031, 0.063, 1)   # #080810

# ── Cerere permisiuni Android ─────────────────────────────────
def _request_android_permissions():
    if not IS_ANDROID:
        return
    try:
        from jnius import autoclass
        PythonActivity = autoclass('org.kivy.android.PythonActivity')
        activity = PythonActivity.mActivity
        perms = ["android.permission.RECORD_AUDIO"]
        to_req = []
        for p in perms:
            try:
                if activity.checkSelfPermission(p) != 0:
                    to_req.append(p)
            except Exception:
                to_req.append(p)
        if to_req:
            activity.requestPermissions(to_req, 1001)
    except Exception as e:
        print(f"[Permissions] Eroare: {e}")

# Culori
C_ACC  = (0.878, 0.176, 0.435, 1)
C_TEAL = (0.000, 0.831, 0.667, 1)
C_GOLD = (0.941, 0.753, 0.251, 1)
C_GRN  = (0.000, 1.000, 0.533, 1)
C_DIM  = (0.251, 0.251, 0.345, 1)
C_BLUE = (0.333, 0.600, 1.000, 1)
C_BG   = (0.031, 0.031, 0.063, 1)
C_PAN  = (0.063, 0.063, 0.118, 1)


def hex2rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))


# ── Widget slider cu label ────────────────────────────────────
class DSPSlider(BoxLayout):
    def __init__(self, label, mn, mx, val, unit='', color=C_TEAL,
                 callback=None, fmt=None, **kw):
        super().__init__(orientation='horizontal', size_hint_y=None,
                         height=dp(44), **kw)
        self._cb   = callback
        self._unit = unit
        self._fmt  = fmt
        self.padding = [dp(8), dp(4), dp(8), dp(4)]
        self.spacing = dp(6)

        # Label stânga
        lbl = Label(text=label, size_hint_x=None, width=dp(110),
                    font_name='data/fonts/RobotoMono-Regular.ttf' if False else 'Roboto',
                    font_size=sp(10), bold=True,
                    color=C_TEAL, halign='left', valign='middle')
        lbl.bind(size=lbl.setter('text_size'))
        self.add_widget(lbl)

        # Slider
        self.slider = Slider(min=mn, max=mx, value=val,
                             size_hint_x=1, cursor_size=(dp(20), dp(20)))
        self.slider.bind(value=self._on_change)
        self.add_widget(self.slider)

        # Valoare dreapta
        self.val_lbl = Label(text=self._fmt_val(val),
                             size_hint_x=None, width=dp(58),
                             font_size=sp(10), bold=True,
                             color=C_ACC, halign='right', valign='middle')
        self.val_lbl.bind(size=self.val_lbl.setter('text_size'))
        self.add_widget(self.val_lbl)

        with self.canvas.before:
            Color(*C_PAN)
            self._rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._upd_rect, size=self._upd_rect)

    def _upd_rect(self, *a):
        self._rect.pos  = self.pos
        self._rect.size = self.size

    def _fmt_val(self, v):
        u = self._unit
        if   u == 'dB':  return f'{v:+.1f}dB'
        elif u == ':1':  return f'{v:.1f}:1'
        elif u == 'ms':  return f'{v:.0f}ms'
        elif u == 'x':   return f'{v:.2f}x'
        elif u == '%':   return f'{int(v)}%'
        else:            return f'{v:.2f}'

    def _on_change(self, inst, val):
        self.val_lbl.text = self._fmt_val(val)
        if self._cb:
            self._cb(val)

    def get(self): return self.slider.value
    def set(self, v): self.slider.value = v


# ── VU Meter ─────────────────────────────────────────────────
class VUMeter(Widget):
    def __init__(self, **kw):
        super().__init__(size_hint_y=None, height=dp(22), **kw)
        self._level = 0.0
        self.bind(pos=self._draw, size=self._draw)

    def update(self, rms_db):
        self._level = max(0.0, min(1.0, (rms_db + 60) / 60))
        self._draw()

    def _draw(self, *a):
        self.canvas.clear()
        with self.canvas:
            # Background
            Color(0.06, 0.06, 0.12, 1)
            Rectangle(pos=self.pos, size=self.size)
            # Level bar
            w = self.width * self._level
            if self._level < 0.7:
                Color(*C_GRN)
            elif self._level < 0.9:
                Color(*C_GOLD)
            else:
                Color(*C_ACC)
            Rectangle(pos=self.pos, size=(w, self.height))
            # Border
            Color(*C_DIM)
            Line(rectangle=(*self.pos, *self.size), width=dp(1))


# ── PEQ Row compact ──────────────────────────────────────────
class PEQRow(BoxLayout):
    """Rând PEQ compact: ON | Freq | Gain | Q"""
    def __init__(self, idx, name, f_def, callback=None, **kw):
        super().__init__(orientation='horizontal', size_hint_y=None,
                         height=dp(44), **kw)
        self.idx = idx; self._cb = callback
        self.freq = f_def; self.gain_db = 0.0
        self.q    = 1.4;   self.enabled = False
        self.padding = [dp(6), dp(2), dp(6), dp(2)]
        self.spacing = dp(4)

        self._on_btn = ToggleButton(text=name, size_hint_x=None, width=dp(72),
                                    font_size=sp(9), bold=True,
                                    background_normal='', background_down='',
                                    background_color=C_PAN, color=C_DIM)
        self._on_btn.bind(on_press=self._tog)
        self.add_widget(self._on_btn)

        for attr, mn, mx, dfl, col, sfx in [
            ('freq',    20, 20000, f_def, C_TEAL, 'Hz'),
            ('gain_db',-12,    12,   0.0, C_ACC,  'dB'),
            ('q',      0.3,   8.0,   1.4, C_BLUE, 'Q'),
        ]:
            sl = Slider(min=mn, max=mx, value=dfl, size_hint_x=1,
                        cursor_size=(dp(18), dp(18)))
            sl._ab = attr
            sl.bind(value=self._changed)
            self.add_widget(sl)
            setattr(self, f'_sl_{attr}', sl)
            fmt = f'{dfl:.0f}{sfx}' if attr == 'freq' else (
                  f'{dfl:+.1f}{sfx}' if attr == 'gain_db' else f'Q{dfl:.1f}')
            lv = Label(text=fmt, size_hint_x=None, width=dp(46),
                       font_size=sp(8), bold=True, color=col,
                       halign='center', valign='middle')
            lv.bind(size=lv.setter('text_size'))
            setattr(self, f'_lv_{attr}', lv)
            self.add_widget(lv)

        with self.canvas.before:
            Color(*C_PAN)
            self._r = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=lambda *_: setattr(self._r, 'pos', self.pos),
                  size=lambda *_: setattr(self._r, 'size', self.size))

    def _tog(self, btn):
        self.enabled = not self.enabled
        self._on_btn.background_color = C_GOLD if self.enabled else C_PAN
        self._on_btn.color = (0.0, 0.0, 0.0, 1) if self.enabled else C_DIM
        if self._cb: self._cb(self.idx)

    def _changed(self, sl, val):
        setattr(self, sl._ab, val)
        lv = getattr(self, f'_lv_{sl._ab}')
        if sl._ab == 'freq':    lv.text = f'{val:.0f}Hz'
        elif sl._ab == 'gain_db': lv.text = f'{val:+.1f}dB'
        else:                   lv.text = f'Q{val:.1f}'
        if self._cb: self._cb(self.idx)


# ── Ecran principal ───────────────────────────────────────────
class MainScreen(BoxLayout):
    def __init__(self, **kw):
        super().__init__(orientation='vertical', **kw)
        self.dsp     = RadioDSP()
        self.stream  = None
        self.running = False
        self._err    = 0
        self._in_list  = []
        self._out_list = []

        self._build()
        Clock.schedule_once(lambda dt: self._refresh_devices(), 0.5)

    # ── Build ────────────────────────────────────────────────
    def _build(self):
        # Header
        hdr = BoxLayout(size_hint_y=None, height=dp(52), padding=dp(10))
        with hdr.canvas.before:
            Color(0.04, 0.04, 0.08, 1)
            self._hdr_rect = Rectangle(pos=hdr.pos, size=hdr.size)
        hdr.bind(pos=lambda i,v: setattr(self._hdr_rect,'pos',v),
                 size=lambda i,v: setattr(self._hdr_rect,'size',v))
        hdr.add_widget(Label(
            text='[color=e02d6f]◉ AUDIOBOOST[/color]  [color=00d4aa]v5  RADIO DSP[/color]',
            markup=True, font_size=sp(16), bold=True, halign='left'))
        self.add_widget(hdr)

        # Scroll area
        sv = ScrollView(do_scroll_x=False)
        self._inner = GridLayout(cols=1, spacing=dp(6), padding=dp(8),
                                 size_hint_y=None)
        self._inner.bind(minimum_height=self._inner.setter('height'))
        sv.add_widget(self._inner)
        self.add_widget(sv)

        self._build_master()
        self._build_presets()
        self._build_devices()
        self._build_dsp_sliders()
        self._build_ambience()
        self._build_start_btn()

    def _section(self, title, color=C_DIM):
        lbl = Label(text=f'  {title}', size_hint_y=None, height=dp(28),
                    font_size=sp(9), bold=True, color=color,
                    halign='left', valign='middle')
        lbl.bind(size=lbl.setter('text_size'))
        with lbl.canvas.before:
            Color(0.07, 0.07, 0.15, 1)
            Rectangle(pos=lbl.pos, size=lbl.size)
        lbl.bind(pos=lambda i,v: None, size=lambda i,v: None)
        self._inner.add_widget(lbl)

    # ── Master volume ────────────────────────────────────────
    def _build_master(self):
        self._section('VOLUM MASTER', C_GOLD)
        self._master_sl = DSPSlider(
            '🔊  MASTER', 0, 200, 100, unit='%', color=C_GOLD,
            callback=self._on_master)
        self._inner.add_widget(self._master_sl)

        self.vu = VUMeter()
        self._inner.add_widget(self.vu)

    def _on_master(self, val):
        self.dsp.set_master(val / 100.0)
        if IS_ANDROID and self.running:
            self._write_dsp_params()

    # ── Preseturi ────────────────────────────────────────────
    def _build_presets(self):
        self._section('PRESETURI', C_TEAL)
        row1 = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(4))
        row2 = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(4))
        names = list(PRESETS.keys())
        mid   = len(names) // 2
        for name in names[:mid]:
            btn = Button(text=name, font_size=sp(9), bold=True,
                         background_color=(0.1, 0.1, 0.18, 1),
                         color=(0.94, 0.75, 0.25, 1))
            btn.bind(on_press=lambda b, n=name: self._load_preset(n))
            row1.add_widget(btn)
        for name in names[mid:]:
            btn = Button(text=name, font_size=sp(9), bold=True,
                         background_color=(0.1, 0.1, 0.18, 1),
                         color=(0.94, 0.75, 0.25, 1))
            btn.bind(on_press=lambda b, n=name: self._load_preset(n))
            row2.add_widget(btn)
        self._inner.add_widget(row1)
        self._inner.add_widget(row2)

    def _load_preset(self, name):
        p = PRESETS[name]
        for key, sl in self._dsp_sliders.items():
            if key in p:
                sl.set(p[key])
        if 'amb_wet'  in p: self._amb_wet.set(p['amb_wet'])
        if 'amb_room' in p: self._amb_disp.set(p['amb_room'])
        if 'sur_str'  in p: self._sur_sl.set(p['sur_str'])
        self._on_sl()
        if IS_ANDROID and self.running:
            self._write_dsp_params()
        self._log(f'Preset: {name}')

    # ── Dispozitive ─────────────────────────────────────────
    def _build_devices(self):
        self._section('DISPOZITIVE', C_BLUE)

        if IS_ANDROID:
            # Android: Output device selector
            out_row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(6))
            out_row.padding = [dp(8), dp(4), dp(8), dp(4)]
            out_row.add_widget(Label(text='OUTPUT:', size_hint_x=None, width=dp(70),
                                     font_size=sp(10), bold=True, color=C_TEAL))
            self._out_spin = Spinner(text='Auto (Default)', values=['Auto (Default)'],
                                     size_hint_x=1, font_size=sp(9),
                                     background_color=(0.1, 0.1, 0.18, 1),
                                     color=C_GOLD)
            self._out_spin.bind(text=self._on_output_device_change)
            out_row.add_widget(self._out_spin)
            self._inner.add_widget(out_row)

            ref_btn = Button(text='↺  Detectează dispozitive',
                             size_hint_y=None, height=dp(36), font_size=sp(10),
                             background_color=(0.1, 0.1, 0.18, 1), color=C_TEAL)
            ref_btn.bind(on_press=lambda b: self._refresh_devices())
            self._inner.add_widget(ref_btn)

            info = Label(
                text='[color=f0c040]Procesare audio SYSTEM-WIDE (Spotify, YouTube, etc.)[/color]\n'
                     'Apasă START → acordă permisiunea de captură audio.\n'
                     'Selectează dispozitivul de ieșire (difuzor/căști).',
                size_hint_y=None, height=dp(50),
                font_size=sp(8), color=C_DIM,
                halign='left', valign='top', markup=True)
            info.bind(size=info.setter('text_size'))
            self._inner.add_widget(info)
        else:
            # Desktop: selecție dispozitive
            in_row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(6))
            in_row.add_widget(Label(text='IN:', size_hint_x=None, width=dp(36),
                                    font_size=sp(10), color=C_TEAL))
            self._in_spin = Spinner(text='— selectează —', values=[],
                                    size_hint_x=1, font_size=sp(9),
                                    background_color=(0.1, 0.1, 0.18, 1))
            in_row.add_widget(self._in_spin)
            self._inner.add_widget(in_row)

            out_row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(6))
            out_row.add_widget(Label(text='OUT:', size_hint_x=None, width=dp(36),
                                     font_size=sp(10), color=C_TEAL))
            self._out_spin = Spinner(text='— selectează —', values=[],
                                     size_hint_x=1, font_size=sp(9),
                                     background_color=(0.1, 0.1, 0.18, 1))
            out_row.add_widget(self._out_spin)
            self._inner.add_widget(out_row)

            ref_btn = Button(text='↺  Reîncarcă dispozitive',
                             size_hint_y=None, height=dp(36), font_size=sp(10),
                             background_color=(0.1, 0.1, 0.18, 1), color=C_TEAL)
            ref_btn.bind(on_press=lambda b: self._refresh_devices())
            self._inner.add_widget(ref_btn)

        # Log
        self._log_lbl = Label(text='AudioBoost v5 — DSP Radio Profesional',
                              size_hint_y=None, height=dp(48),
                              font_size=sp(8), color=C_DIM,
                              halign='left', valign='top', markup=True)
        self._log_lbl.bind(size=self._log_lbl.setter('text_size'))
        self._inner.add_widget(self._log_lbl)

    def _log(self, msg):
        prev = self._log_lbl.text.split('\n')[:2]
        self._log_lbl.text = msg + '\n' + '\n'.join(prev)

    def _refresh_devices(self):
        if IS_ANDROID:
            self._detect_android_output_devices()
            return
        try:
            import sounddevice as sd
            devs = sd.query_devices()
            self._in_list  = [(i, d) for i, d in enumerate(devs) if d['max_input_channels']  > 0]
            self._out_list = [(i, d) for i, d in enumerate(devs) if d['max_output_channels'] > 0]
            self._in_spin.values  = [f"[{i}] {d['name']}" for i, d in self._in_list]
            self._out_spin.values = [f"[{i}] {d['name']}" for i, d in self._out_list]
            # Auto-select CABLE / default
            for idx, (i, d) in enumerate(self._in_list):
                if any(k in d['name'].lower() for k in ['cable output','stereo mix']):
                    self._in_spin.text = self._in_spin.values[idx]; break
            else:
                if self._in_spin.values: self._in_spin.text = self._in_spin.values[0]
            if self._out_spin.values: self._out_spin.text = self._out_spin.values[0]
            self._log(f'{len(self._in_list)} IN, {len(self._out_list)} OUT găsite')
        except Exception as e:
            self._log(f'Eroare dispozitive: {e}')

    def _get_device_idx(self, spin, lst):
        txt = spin.text
        for idx, (i, d) in enumerate(lst):
            if spin.values and idx < len(spin.values) and spin.values[idx] == txt:
                return i
        return None

    def _detect_android_output_devices(self):
        """Detectează dispozitivele de ieșire audio pe Android."""
        try:
            from jnius import autoclass
            Context = autoclass('android.content.Context')
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            activity = PythonActivity.mActivity
            AudioManager = autoclass('android.media.AudioManager')
            am = activity.getSystemService(Context.AUDIO_SERVICE)

            GET_DEVICES_OUTPUTS = 2  # AudioManager.GET_DEVICES_OUTPUTS
            devices = am.getDevices(GET_DEVICES_OUTPUTS)

            self._android_out_devices = []
            names = ['Auto (Default)']

            # Device type mapping
            TYPE_MAP = {
                1: 'Earpiece',
                2: 'Speaker',
                3: 'Wired Headset',
                4: 'Wired Headphones',
                7: 'Bluetooth SCO',
                8: 'Bluetooth A2DP',
                13: 'USB Device',
                14: 'USB Headset',
                22: 'USB Accessory',
            }

            for i in range(devices.length):
                d = devices[i]
                dtype = int(d.getType())
                dname = d.getProductName().toString()
                did = int(d.getId())
                type_name = TYPE_MAP.get(dtype, f'Type {dtype}')
                label = f'{type_name} ({dname})' if dname else type_name
                names.append(label)
                self._android_out_devices.append({
                    'type': dtype, 'name': dname, 'id': did, 'label': label,
                    'device': d
                })

            self._out_spin.values = names
            self._log(f'Android: {len(self._android_out_devices)} dispozitive OUT găsite')
        except Exception as e:
            self._log(f'Eroare detectare dispozitive: {e}')
            self._android_out_devices = []

    def _on_output_device_change(self, spinner, text):
        """Când utilizatorul selectează un dispozitiv de ieșire."""
        if not IS_ANDROID:
            return
        if text == 'Auto (Default)':
            self._selected_output_device = None
        elif hasattr(self, '_android_out_devices'):
            for dev in self._android_out_devices:
                if dev['label'] == text:
                    self._selected_output_device = dev
                    break
        # Write to SharedPreferences if service is running
        if self.running:
            self._write_dsp_params()

    def _write_dsp_params(self):
        """Scrie parametrii DSP în SharedPreferences pentru service."""
        if not IS_ANDROID:
            return
        try:
            import json
            sv = self._dsp_sliders
            if not sv or not hasattr(self, '_amb_wet'):
                return
            params = {
                'master': self.dsp.get_master(),
                'in_db': sv['in_db'].get(), 'bd': sv['bd'].get(),
                'td': sv['td'].get(), 'pd': sv['pd'].get(),
                'ex': sv['ex'].get(), 'thr': sv['thr'].get(),
                'rat': sv['rat'].get(), 'mkup': sv['mkup'].get(),
                'pmix': sv['pmix'].get(), 'sw': sv['sw'].get(),
                'haas': sv['haas'].get(), 'od': sv['od'].get(),
                'ds': sv['ds'].get(), 'up': sv['up'].get(),
                'amb_wet': self._amb_wet.get(),
                'amb_room': self._amb_disp.get(),
                'amb_damp': self._amb_damp.get(),
                'amb_pre': self._amb_pre.get(),
                'sur_str': self._sur_sl.get(),
            }
            # Output device
            if hasattr(self, '_selected_output_device') and self._selected_output_device:
                params['output_device_id'] = self._selected_output_device['id']
            else:
                params['output_device_id'] = -1
            # PEQ bands
            peq = []
            for row in self._peq_rows:
                peq.append({
                    'freq': row.freq, 'gain_db': row.gain_db,
                    'q': row.q, 'enabled': row.enabled
                })
            params['peq'] = peq

            from jnius import autoclass
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            activity = PythonActivity.mActivity
            sp = activity.getSharedPreferences('audioboost_prefs', 0)
            editor = sp.edit()
            editor.putString('dsp_params', json.dumps(params))
            editor.apply()
        except Exception as e:
            print(f'[IPC] Write error: {e}')

    # ── DSP Sliders ─────────────────────────────────────────
    def _build_dsp_sliders(self):
        self._section('PROCESARE DSP', C_TEAL)
        sliders_def = [
            ('in_db', 'TRIM INTRARE',  -12,  12,   'dB',   0.0,  C_TEAL),
            ('bd',    'BASS',           -3,  10,   'dB',   5.0,  C_GOLD),
            ('td',    'TREBLE',         -3,  10,   'dB',   3.5,  C_GOLD),
            ('pd',    'PREZENȚĂ',       -3,  10,   'dB',   4.0,  C_GOLD),
            ('ex',    'EXCITER',         0,   1,    '',    0.35,  C_TEAL),
            ('thr',   'PRAG COMP',     -40,  -6,  'dB', -22.0,   C_ACC),
            ('rat',   'RATIO',           1,  12,  ':1',    5.0,  C_ACC),
            ('mkup',  'MAKEUP GAIN',     0,  15,  'dB',    8.0,  C_GRN),
            ('pmix',  'PARALLEL MIX',    0,   1,   '',    0.40,  C_TEAL),
            ('sw',    'STEREO WIDTH',  1.0, 3.0,   'x',   1.55,  C_BLUE),
            ('haas',  'HAAS DELAY',    0.0,30.0,  'ms',  12.0,   C_BLUE),
            ('ds',    'DE-ESSER',       0,  12,  'dB',    4.0,  C_TEAL),
            ('up',    'HF SHIMMER',     0,   1,   '',    0.30,  C_TEAL),
            ('od',    'VOL OUT DSP',   -6,   6,  'dB',    1.0,  C_GRN),
        ]
        self._dsp_sliders = {}
        for key, lbl, mn, mx, unit, dflt, col in sliders_def:
            sl = DSPSlider(lbl, mn, mx, dflt, unit=unit, color=col,
                           callback=lambda v: self._on_sl())
            self._inner.add_widget(sl)
            self._dsp_sliders[key] = sl

    def _on_sl(self):
        sv = self._dsp_sliders
        if not sv:
            return
        if not hasattr(self, '_amb_wet'):
            return
        self.dsp.update(
            sv['in_db'].get(), sv['bd'].get(), sv['td'].get(),
            sv['pd'].get(),    sv['ex'].get(), sv['thr'].get(),
            sv['rat'].get(),   sv['mkup'].get(),sv['pmix'].get(),
            sv['sw'].get(),    sv['haas'].get(),sv['od'].get(),
            ds=sv['ds'].get(), up=sv['up'].get(),
            amb_wet=self._amb_wet.get(), amb_room=self._amb_disp.get(),
            amb_damp=self._amb_damp.get(), amb_pre=self._amb_pre.get(),
            sur_str=self._sur_sl.get()
        )
        if IS_ANDROID and self.running:
            self._write_dsp_params()

    # ── Ambience + 3D ────────────────────────────────────────
    def _build_ambience(self):
        self._section('AMBIENCE + 3D SURROUND', C_BLUE)
        self._amb_wet  = DSPSlider('AMBIENCE',     0.0, 1.0,  0.0, callback=lambda v: self._on_sl())
        self._amb_disp = DSPSlider('ROOM SIZE',    0.1, 0.9,  0.5, callback=lambda v: self._on_sl())
        self._amb_damp = DSPSlider('DAMPING',      0.0, 1.0, 0.45, callback=lambda v: self._on_sl())
        self._amb_pre  = DSPSlider('PRE-DELAY ms', 0.0,40.0, 15.0, unit='ms', callback=lambda v: self._on_sl())
        self._sur_sl   = DSPSlider('3D SURROUND',  0.0, 1.0,  0.0, callback=lambda v: self._on_sl())
        for w in [self._amb_wet, self._amb_disp, self._amb_damp, self._amb_pre, self._sur_sl]:
            self._inner.add_widget(w)

        # ── Parametric EQ 5 benzi ─────────────────────────────
        self._section('PARAMETRIC EQ  (5 benzi)', C_GOLD)

        # EQ Preset dropdown
        eq_row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(6))
        eq_row.padding = [dp(8), dp(4), dp(8), dp(4)]
        eq_row.add_widget(Label(text='PRESET:', size_hint_x=None, width=dp(70),
                                font_size=sp(10), bold=True, color=C_TEAL,
                                halign='left', valign='middle'))
        self._eq_spinner = Spinner(
            text='Flat (Reset)',
            values=list(EQ_PRESETS.keys()),
            size_hint_x=1, font_size=sp(10),
            background_color=(0.1, 0.1, 0.18, 1),
            color=C_GOLD)
        self._eq_spinner.bind(text=self._on_eq_preset)
        eq_row.add_widget(self._eq_spinner)
        self._inner.add_widget(eq_row)

        PEQ_DEFS = [
            ('Low',      80.0),  ('Low-Mid', 250.0),
            ('Mid',    1000.0),  ('High-Mid',4000.0),
            ('High',  12000.0),
        ]
        self._peq_rows = []
        for i, (name, f_def) in enumerate(PEQ_DEFS):
            row = PEQRow(i, name, f_def, callback=self._on_peq)
            self._inner.add_widget(row)
            self._peq_rows.append(row)

    def _on_peq(self, idx):
        row = self._peq_rows[idx]
        self.dsp.update_peq(idx, row.freq, row.gain_db, row.q, row.enabled)
        if IS_ANDROID and self.running:
            self._write_dsp_params()

    def _on_eq_preset(self, spinner, text):
        """Aplică un preset EQ pe cele 5 benzi parametrice."""
        if text not in EQ_PRESETS:
            return
        settings = EQ_PRESETS[text]
        for i, (freq, gain, q, enabled) in enumerate(settings):
            if i < len(self._peq_rows):
                row = self._peq_rows[i]
                row._sl_freq.value = freq
                row._sl_gain_db.value = gain
                row._sl_q.value = q
                if enabled != row.enabled:
                    row._tog(row._on_btn)
                row.freq = freq
                row.gain_db = gain
                row.q = q
                row.enabled = enabled
                # Update labels
                row._lv_freq.text = f'{freq:.0f}Hz'
                row._lv_gain_db.text = f'{gain:+.1f}dB'
                row._lv_q.text = f'Q{q:.1f}'
                # Update DSP
                self.dsp.update_peq(i, freq, gain, q, enabled)
        if IS_ANDROID and self.running:
            self._write_dsp_params()
        self._log(f'EQ Preset: {text}')

    # ── Start / Stop ─────────────────────────────────────────
    def _build_start_btn(self):
        self._start_btn = ToggleButton(
            text='▶  PORNEȘTE', size_hint_y=None, height=dp(64),
            font_size=sp(14), bold=True,
            background_color=(0.878, 0.176, 0.435, 1),
            background_normal='', background_down='',
            color=(1, 1, 1, 1))
        self._start_btn.bind(on_press=self._toggle)
        self._inner.add_widget(self._start_btn)

    def _toggle(self, btn):
        if not self.running:
            self._start()
        else:
            self._stop()

    def _start(self):
        if IS_ANDROID:
            self._start_android()
        elif SD_OK:
            self._start_desktop()
        else:
            self._log('sounddevice lipsă — instalează: pip install sounddevice')

    def _start_android(self):
        """Pornește procesarea audio system-wide prin MediaProjection service."""
        try:
            from jnius import autoclass
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            activity = PythonActivity.mActivity
            Context = autoclass('android.content.Context')

            # 1. Detectează dispozitive output
            if not hasattr(self, '_android_out_devices'):
                self._detect_android_output_devices()

            # 2. Scrie parametrii DSP inițiali
            self.dsp = RadioDSP(sr=44100, ch=CHANNELS)
            self._on_master(self._master_sl.get())
            self._on_sl()
            self._write_dsp_params()

            # 3. Request MediaProjection permission via startActivityForResult
            REQUEST_CODE_MP = 1001

            from android.activity import on_activity_result
            main_self = self

            def _on_mp_result(requestCode, resultCode, data):
                if requestCode != REQUEST_CODE_MP:
                    return
                if resultCode == -1 and data is not None:  # RESULT_OK
                    Clock.schedule_once(
                        lambda dt: main_self._start_service_with_projection(
                            resultCode, data), 0)
                else:
                    Clock.schedule_once(
                        lambda dt: main_self._log(
                            'Permisiune MediaProjection refuzată'), 0)
                    Clock.schedule_once(
                        lambda dt: setattr(main_self._start_btn, 'state', 'normal'), 0)

            on_activity_result(REQUEST_CODE_MP, _on_mp_result)

            MProjectionManager = autoclass('android.media.projection.MediaProjectionManager')
            mp_mgr = activity.getSystemService(Context.MEDIA_PROJECTION_SERVICE)
            intent = mp_mgr.createScreenCaptureIntent()
            activity.startActivityForResult(intent, REQUEST_CODE_MP)
            self._log('Aștept permisiunea MediaProjection...')

        except Exception as e:
            self._log(f'EROARE Android: {e}')
            self._start_btn.state = 'normal'

    def _start_service_with_projection(self, result_code, result_data):
        """Pornește service-ul cu MediaProjection data."""
        try:
            from jnius import autoclass
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            activity = PythonActivity.mActivity
            Intent = autoclass('android.content.Intent')
            Context = autoclass('android.content.Context')

            # Create intent for the service
            service_cls = autoclass('org.audioboost.AudioBoost')
            intent = Intent(activity, service_cls)
            intent.putExtra('resultCode', result_code)
            # Pass the MediaProjection result data (Intent) as Parcelable
            intent.putExtra('resultData', result_data)

            # Start foreground service
            activity.startForegroundService(intent)

            self.running = True
            self._start_btn.text = '⏹  OPREȘTE'
            self._log('▶ Procesare audio system-wide pornită!')
            self._log('  Redă muzică în Spotify/YouTube pentru a auzi efectul')
            Clock.schedule_interval(self._vu_tick, 0.5)
        except Exception as e:
            self._log(f'EROARE pornire service: {e}')
            self._start_btn.state = 'normal'

    def _start_desktop(self):
        """Audio desktop prin sounddevice."""
        in_idx  = self._get_device_idx(self._in_spin,  self._in_list)
        out_idx = self._get_device_idx(self._out_spin, self._out_list)
        if in_idx is None or out_idx is None:
            self._log('Selectează dispozitivele IN și OUT!'); return
        try:
            ind = sd.query_devices(in_idx)
            oud = sd.query_devices(out_idx)
            sr  = int(ind['default_samplerate'])
            ic  = min(int(ind['max_input_channels']),  CHANNELS)
            oc  = min(int(oud['max_output_channels']), CHANNELS)
            self.dsp = RadioDSP(sr=sr, ch=ic)
            self._on_master(self._master_sl.get())
            self._on_sl()
            self._err = 0
            self.stream = sd.Stream(
                samplerate=sr, blocksize=BLOCKSIZE,
                channels=(ic, oc), dtype='float32',
                device=(in_idx, out_idx),
                callback=self._cb, latency='high')
            self.stream.start()
            self.running = True
            self._start_btn.text = '⏹  OPREȘTE'
            self._log(f'▶ {ind["name"]} → {oud["name"]}  {sr}Hz')
            Clock.schedule_interval(self._vu_tick, 0.05)
        except Exception as e:
            self._log(f'EROARE: {e}')
            self._start_btn.state = 'normal'

    def _stop(self):
        Clock.unschedule(self._vu_tick)
        if IS_ANDROID:
            self._stop_android()
        else:
            if self.stream:
                try: self.stream.stop(); self.stream.close()
                except: pass
                self.stream = None
        self.running = False
        self._start_btn.text = '▶  PORNEȘTE'
        self._start_btn.state = 'normal'
        self.vu.update(-60)
        self._log('⏸ Oprit.')

    def _stop_android(self):
        """Oprește service-ul Android."""
        try:
            from jnius import autoclass
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            activity = PythonActivity.mActivity
            Intent = autoclass('android.content.Intent')
            service_cls = autoclass('org.audioboost.AudioBoost')
            intent = Intent(activity, service_cls)
            activity.stopService(intent)
        except Exception as e:
            self._log(f'Eroare oprire service: {e}')

    def _cb(self, indata, outdata, frames, time_info, status):
        try:
            self._last_rms = float(np.sqrt(np.mean(indata**2)) + 1e-10)
            proc = self.dsp.process(indata.copy())
            proc = proc * self.dsp.get_master()
            no   = outdata.shape[1]
            if proc.ndim == 1: proc = proc[:, np.newaxis]
            np_  = proc.shape[1]
            if np_ >= no:   outdata[:] = proc[:, :no]
            else:
                outdata[:, :np_] = proc
                outdata[:, np_:] = proc[:, -1:]
        except Exception:
            outdata[:] = 0
            self._err += 1

    def _vu_tick(self, dt):
        rms = getattr(self, '_last_rms', 1e-10)
        db  = 20 * math.log10(rms + 1e-10)
        self.vu.update(db)


class AudioBoostApp(KivyApp):
    def build(self):
        self.title = 'AudioBoost v5'
        screen = MainScreen()
        # Fix _cb rms attribute
        screen._last_rms = 1e-10
        return screen

    def on_start(self):
        if IS_ANDROID:
            Clock.schedule_once(lambda dt: _request_android_permissions(), 2.0)


def main():
    try:
        AudioBoostApp().run()
    except Exception as e:
        # Afișează eroarea pe ecran în loc de crash
        import traceback, sys
        tb = traceback.format_exc()
        print(f"[FATAL] {tb}")
        # Scrie și în logcat pentru Android
        try:
            from jnius import autoclass
            Log = autoclass('android.util.Log')
            Log.e('AudioBoost', f'FATAL: {tb}')
        except:
            pass
        try:
            from kivy.app import App as _A
            from kivy.uix.label import Label
            from kivy.uix.boxlayout import BoxLayout
            from kivy.core.window import Window
            class ErrApp(_A):
                def build(self):
                    self.title = 'AudioBoost - EROARE'
                    Window.clearcolor = (0.03, 0.03, 0.06, 1)
                    bl = BoxLayout(orientation='vertical', padding=20)
                    bl.add_widget(Label(
                        text=f'EROARE:\\n{e}\\n\\n{tb[-500:]}',
                        font_size='12sp', halign='left', valign='top',
                        color=(1, 0.3, 0.3, 1), text_size=(Window.width-40, None)))
                    return bl
            ErrApp().run()
        except:
            pass


if __name__ == '__main__':
    try:
        main()
    except ImportError as e:
        # Eroare de import — logcat + ecran de eroare
        import traceback, sys
        tb = traceback.format_exc()
        print(f"[IMPORT ERROR] {tb}")
        try:
            from kivy.app import App as _A
            from kivy.uix.label import Label
            from kivy.uix.boxlayout import BoxLayout
            from kivy.core.window import Window
            class ErrApp(_A):
                def build(self):
                    self.title = 'AudioBoost - Eroare Import'
                    Window.clearcolor = (0.03, 0.03, 0.06, 1)
                    bl = BoxLayout(orientation='vertical', padding=20)
                    bl.add_widget(Label(
                        text=f'Eroare import:\\n{e}\\n\\n{tb[-800:]}',
                        font_size='12sp', halign='left', valign='top',
                        color=(1, 0.3, 0.3, 1), text_size=(Window.width-40, None)))
                    return bl
            ErrApp().run()
        except:
            sys.exit(1)
