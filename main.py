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
from scipy.signal import butter, sosfilt, sosfilt_zi, lfilter

# ══════════════════════════════════════════════════════════════
#   DSP — identic cu versiunea desktop
# ══════════════════════════════════════════════════════════════

BLOCKSIZE = 1024
CHANNELS  = 2

def _mc(sos, x, zi):
    ch = x.shape[1]; y = np.empty_like(x); zi_new = np.empty_like(zi)
    for c in range(ch):
        y[:, c], zi_new[:, c, :] = sosfilt(sos, x[:, c], zi=zi[:, c, :])
    return y, zi_new


class BinauralSurround3D:
    def __init__(self, sr):
        self.sr = sr
        b, a = butter(1, 800, 'low', fs=sr)
        self.b, self.a = b, a
        self.zi = [np.zeros(max(len(a), len(b)) - 1),
                   np.zeros(max(len(a), len(b)) - 1)]
        self.delay_samp = max(1, int(sr * 0.00032))
        self.history = np.zeros((self.delay_samp, 2))

    def process(self, x, strength):
        if strength < 0.01 or x.shape[1] < 2:
            return x
        fx = np.empty_like(x)
        fx[:, 0], self.zi[0] = lfilter(self.b, self.a, x[:, 0], zi=self.zi[0])
        fx[:, 1], self.zi[1] = lfilter(self.b, self.a, x[:, 1], zi=self.zi[1])
        delayed = np.concatenate([self.history, fx])
        cf = delayed[:-self.delay_samp]
        self.history = delayed[-self.delay_samp:]
        out = np.empty_like(x)
        out[:, 0] = x[:, 0] + cf[:, 1] * (strength * 0.55)
        out[:, 1] = x[:, 1] + cf[:, 0] * (strength * 0.55)
        return out / (1.0 + strength * 0.3)


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

    def process(self, x, amount, size):
        if amount < 0.01: return x
        n = x.shape[0]; N = self._N
        aL, self._hp_zi[0] = lfilter(self._hp_b, self._hp_a, x[:,0], zi=self._hp_zi[0])
        aR, self._hp_zi[1] = lfilter(self._hp_b, self._hp_a, x[:,1], zi=self._hp_zi[1])
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
            d  = max(1, int(self.sr * ms * sc / 1000))
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
        self.surround_str=0.0
        self._lock=threading.Lock(); self._master_lin=1.0
        self._build_filters()

    def _build_filters(self):
        sr=self.sr; ch=self.ch
        def sos(ftype, freq):
            s  = butter(2, freq, ftype, fs=sr, output='sos')
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
            sur_str=self.surround_str
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
        x=self._ambience.process(x,amb_wet,amb_room)
        x=self._surround.process(x,sur_str)
        if od!=0.0: x*=self._lin(od)
        x=self._limiter_vec(x)
        return x.astype(np.float32)

    def set_master(self,v): self._master_lin=float(v)
    def get_master(self):   return self._master_lin

    def update(self, in_db,bd,td,pd,ex,thr,rat,mkup,pmix,sw,haas,od,
               ds=0.0, up=0.0, amb_wet=0.0, amb_room=0.5, sur_str=0.0):
        with self._lock:
            self.input_db=in_db; self.bass_db=bd; self.treble_db=td
            self.presence_db=pd; self.exciter=ex; self.thresh_db=thr
            self.ratio=rat; self.makeup_db=mkup; self.parallel_mix=pmix
            self.stereo_w=sw; self.haas_ms=haas; self.output_db=od
            self.deesser_db=ds; self.upscale=up
            self.ambience_wet=amb_wet; self.ambience_room=amb_room
            self.surround_str=sur_str


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
        self._log(f'Preset: {name}')

    # ── Dispozitive ─────────────────────────────────────────
    def _build_devices(self):
        self._section('DISPOZITIVE', C_BLUE)

        # IN spinner
        in_row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(6))
        in_row.add_widget(Label(text='IN:', size_hint_x=None, width=dp(36),
                                font_size=sp(10), color=C_TEAL))
        self._in_spin = Spinner(text='— selectează —', values=[],
                                size_hint_x=1, font_size=sp(9),
                                background_color=(0.1, 0.1, 0.18, 1))
        in_row.add_widget(self._in_spin)
        self._inner.add_widget(in_row)

        # OUT spinner
        out_row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(6))
        out_row.add_widget(Label(text='OUT:', size_hint_x=None, width=dp(36),
                                 font_size=sp(10), color=C_TEAL))
        self._out_spin = Spinner(text='— selectează —', values=[],
                                 size_hint_x=1, font_size=sp(9),
                                 background_color=(0.1, 0.1, 0.18, 1))
        out_row.add_widget(self._out_spin)
        self._inner.add_widget(out_row)

        # Refresh btn
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
        self.dsp.update(
            sv['in_db'].get(), sv['bd'].get(), sv['td'].get(),
            sv['pd'].get(),    sv['ex'].get(), sv['thr'].get(),
            sv['rat'].get(),   sv['mkup'].get(),sv['pmix'].get(),
            sv['sw'].get(),    sv['haas'].get(),sv['od'].get(),
            ds=sv['ds'].get(), up=sv['up'].get(),
            amb_wet=self._amb_wet.get(), amb_room=self._amb_disp.get(),
            sur_str=self._sur_sl.get()
        )

    # ── Ambience + 3D ────────────────────────────────────────
    def _build_ambience(self):
        self._section('AMBIENCE + 3D SURROUND', C_BLUE)
        self._amb_wet  = DSPSlider('AMBIENCE',    0.0, 1.0, 0.0, callback=lambda v: self._on_sl())
        self._amb_disp = DSPSlider('DISPERSIE',   0.1, 0.9, 0.5, callback=lambda v: self._on_sl())
        self._sur_sl   = DSPSlider('3D SURROUND', 0.0, 2.0, 0.0, callback=lambda v: self._on_sl())
        self._inner.add_widget(self._amb_wet)
        self._inner.add_widget(self._amb_disp)
        self._inner.add_widget(self._sur_sl)

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
        import sounddevice as sd
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
            # VU meter clock
            Clock.schedule_interval(self._vu_tick, 0.05)
        except Exception as e:
            self._log(f'EROARE: {e}')
            self._start_btn.state = 'normal'

    def _stop(self):
        Clock.unschedule(self._vu_tick)
        if self.stream:
            try: self.stream.stop(); self.stream.close()
            except: pass
            self.stream = None
        self.running = False
        self._start_btn.text = '▶  PORNEȘTE'
        self._start_btn.state = 'normal'
        self.vu.update(-60)
        self._log('⏸ Oprit.')

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


def main():
    AudioBoostApp().run()


if __name__ == '__main__':
    main()
