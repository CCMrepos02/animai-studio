#!/usr/bin/env python3
"""AnimAI Local Server v2 — All tools, port 7474
Run with venv python: /Users/cm/Documents/felix-v2/SadTalker/venv_sadtalker/bin/python3 server.py
"""
import os, io, json, time, tempfile, mimetypes, traceback, subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler
from email import message_from_bytes
from email.policy import HTTP
from pathlib import Path

FFMPEG  = "/opt/homebrew/bin/ffmpeg"
VENV_PY = "/Users/cm/Documents/felix-v2/SadTalker/venv_sadtalker/bin/python3"
INSWAP  = "/Users/cm/Documents/felix-v2/julia-project/models/inswapper_128.onnx"
OUTPUT  = os.path.expanduser("~/Desktop/photo-animator/output")
os.makedirs(OUTPUT, exist_ok=True)

QUALITY_MAP = {
    "high":"1920x1080","medium":"1280x720","low":"854x480",
    "youtube":"1920x1080","ytshorts":"1080x1920"
}

# ─── Lazy AI imports ──────────────────────────────────────────────────────────
_face_app = _face_swapper = None

def get_face_tools():
    global _face_app, _face_swapper
    if _face_app is None:
        from insightface.app import FaceAnalysis
        import insightface
        _face_app = FaceAnalysis(name="buffalo_l")
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
        _face_swapper = insightface.model_zoo.get_model(
            INSWAP, providers=["CPUExecutionProvider"])
    return _face_app, _face_swapper

# ─── Animation engine ─────────────────────────────────────────────────────────
def make_animation(src, style, duration, quality, fmt, grade, vignette, grain, sharpen):
    from PIL import Image
    size_str = QUALITY_MAP.get(quality, "1280x720")
    w, h = [int(x) for x in size_str.split("x")]
    if   fmt == "square":   w = h = min(w, h)
    elif fmt == "vertical": w = int(h * 9 // 16)
    elif fmt == "youtube":  w, h = 1920, 1080
    elif fmt == "ytshorts": w, h = 1080, 1920

    fps, dur, BIG = 30, int(duration), max(w, h) + 400
    N   = dur * fps
    out = tempfile.mktemp(suffix=".mp4", dir="/tmp")

    S = {
      # ── Original 12 ──────────────────────────────────────────────────────────
      "kenburns":   f"scale={BIG}:{BIG},zoompan=z='min(1.0+0.12*on/{N},1.12)':x='iw/2-(iw/zoom/2)':y='max(0,ih/2-(ih/zoom/2)-20*on/{N})':d=1:s={w}x{h}:fps={fps}",
      "baywatch":   f"scale={BIG}:{BIG},zoompan=z='if(lte(on,{N//3}),1.22-0.18*on/({N//3}),if(lte(on,{N*2//3}),1.04,1.04+0.12*(on-{N*2//3})/({N//3})))':x='iw/2-(iw/zoom/2)':y='max(0,ih*0.06-ih/(zoom*2))':d=1:s={w}x{h}:fps={fps}",
      "drift":      f"scale={BIG}:{BIG},zoompan=z='1.06':x='iw*0.08*on/{N}':y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps={fps}",
      "pulse":      f"scale={BIG}:{BIG},zoompan=z='1.0+0.04*sin(2*3.14159*on/({fps*2}))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps={fps}",
      "zoom":       f"scale={BIG}:{BIG},zoompan=z='min(1.0+0.20*on/{N},1.20)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps={fps}",
      "float":      f"scale={BIG}:{BIG},zoompan=z='1.04+0.02*sin(2*3.14159*on/({fps*4}))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)+30*sin(2*3.14159*on/({fps*4}))':d=1:s={w}x{h}:fps={fps}",
      "cinema":     f"scale={BIG}:{BIG},zoompan=z='1.0+0.10*on/{N}':x='iw/2-(iw/zoom/2)':y='max(0,ih*0.15-ih/(zoom*2)-20*on/{N})':d=1:s={w}x{h}:fps={fps}",
      "warmglow":   f"scale={BIG}:{BIG},zoompan=z='1.05+0.04*sin(2*3.14159*on/({fps*3}))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps={fps},eq=brightness=0.05:saturation=1.3:gamma_r=1.1:gamma_b=0.92",
      "glitch":     f"scale={BIG}:{BIG},zoompan=z='1.05':x='iw/2-(iw/zoom/2)+if(eq(mod(on,15),0),20,0)':y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps={fps}",
      "vortex":     f"scale={BIG}:{BIG},zoompan=z='1.12':x='iw/2-(iw/zoom/2)+15*sin(2*3.14159*on/({fps*6}))':y='ih/2-(ih/zoom/2)+15*cos(2*3.14159*on/({fps*6}))':d=1:s={w}x{h}:fps={fps}",
      "reveal":     f"scale={BIG}:{BIG},zoompan=z='1.0+0.08*on/{N}':x='iw*0.05*on/{N}':y='max(0,ih/2-ih/(zoom*2))':d=1:s={w}x{h}:fps={fps}",
      "neon":       f"scale={BIG}:{BIG},zoompan=z='1.04+0.03*sin(2*3.14159*on/{fps})':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps={fps},hue=s=1.8,eq=contrast=1.2",
      # ── New 18 ───────────────────────────────────────────────────────────────
      "parallax":   f"scale={BIG}:{BIG},zoompan=z='1.10':x='iw/2-(iw/zoom/2)+iw*0.06*sin(2*3.14159*on/{N})':y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps={fps}",
      "matrix":     f"scale={BIG}:{BIG},zoompan=z='min(1.0+0.18*on/{N},1.18)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps={fps},hue=h=120:s=2.0,eq=contrast=1.3:brightness=-0.08",
      "bokeh":      f"scale={BIG}:{BIG},zoompan=z='1.08+0.06*sin(2*3.14159*on/({fps*4}))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps={fps},unsharp=lx=5:ly=5:la=-0.4",
      "holographic":f"scale={BIG}:{BIG},zoompan=z='1.06':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps={fps},hue=h='t*25':s=1.6,eq=contrast=1.1",
      "drone":      f"scale={BIG}:{BIG},zoompan=z='max(1.0,1.28-0.22*on/{N})':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps={fps},eq=contrast=1.08:saturation=1.1",
      "timelapse":  f"scale={BIG}:{BIG},zoompan=z='min(1.0+0.38*on/{N},1.38)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps={fps},eq=saturation=1.4:contrast=1.15",
      "noir":       f"scale={BIG}:{BIG},zoompan=z='1.0+0.09*on/{N}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps={fps},eq=contrast=1.6:saturation=0.0:brightness=-0.04",
      "chromatic":  f"scale={BIG}:{BIG},zoompan=z='1.05':x='iw/2-(iw/zoom/2)+if(eq(mod(on,20),0),7,if(eq(mod(on,20),1),-7,0))':y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps={fps},hue=s=1.3",
      "vhs":        f"scale={BIG}:{BIG},zoompan=z='1.03':x='iw/2-(iw/zoom/2)+if(eq(mod(on,25),0),9,0)':y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps={fps},eq=saturation=0.25:contrast=1.15:brightness=-0.05",
      "starfield":  f"scale={BIG}:{BIG},zoompan=z='min(1.0+0.55*on/{N},1.55)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps={fps},eq=brightness=-0.1:contrast=1.4",
      "firestorm":  f"scale={BIG}:{BIG},zoompan=z='1.06+0.03*sin(2*3.14159*on/({fps*2}))':x='iw/2-(iw/zoom/2)+10*sin(2*3.14159*on/({fps*3}))':y='ih/2-(ih/zoom/2)+12*sin(2*3.14159*on/({fps*2}))':d=1:s={w}x{h}:fps={fps},eq=brightness=0.08:saturation=1.6:gamma_r=1.25:gamma_b=0.72",
      "icecrystal": f"scale={BIG}:{BIG},zoompan=z='1.0+0.12*on/{N}':x='iw/2-(iw/zoom/2)+4*sin(2*3.14159*on/({fps*8}))':y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps={fps},eq=contrast=1.35:saturation=0.65:gamma_b=1.35",
      "sunrise":    f"scale={BIG}:{BIG},zoompan=z='1.02+0.07*on/{N}':x='iw/2-(iw/zoom/2)':y='max(0,ih*0.08-ih/(zoom*2)-8*on/{N})':d=1:s={w}x{h}:fps={fps},eq=saturation=1.25:gamma_r=1.08",
      "earthquake": f"scale={BIG}:{BIG},zoompan=z='1.06':x='iw/2-(iw/zoom/2)+if(lt(mod(on,8),4),7,-7)':y='ih/2-(ih/zoom/2)+if(lt(mod(on,6),3),5,-5)':d=1:s={w}x{h}:fps={fps}",
      "liquid":     f"scale={BIG}:{BIG},zoompan=z='1.08+0.04*sin(2*3.14159*on/({fps*3}))':x='iw/2-(iw/zoom/2)+22*sin(2*3.14159*on/({fps*4}))':y='ih/2-(ih/zoom/2)+16*cos(2*3.14159*on/({fps*3}))':d=1:s={w}x{h}:fps={fps}",
      "tiltshift":  f"scale={BIG}:{BIG},zoompan=z='1.05+0.05*on/{N}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps={fps},vignette=PI/2.2",
      "letterbox":  f"scale={BIG}:{BIG},zoompan=z='1.0+0.14*on/{N}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)-18*on/{N}':d=1:s={w}x{h}:fps={fps},eq=contrast=1.12",
      "splitreveal":f"scale={BIG}:{BIG},zoompan=z='1.07':x='iw*0.07*on/{N}':y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps={fps},eq=contrast=1.08:saturation=1.1",
    }
    filters = [S.get(style, S["kenburns"])]
    if grade   == "1" and "eq=" not in filters[0]: filters.append("eq=contrast=1.10:saturation=1.15")
    if sharpen == "1": filters.append("unsharp=5:5:0.8:5:5:0")
    if vignette == "1": filters.append("vignette=PI/3.5")
    filters.append("format=yuv420p")

    yt_enc = fmt in ("youtube","ytshorts")
    enc = ["-b:v","8000k","-maxrate","10000k","-bufsize","20000k","-profile:v","high","-level","4.1"] if yt_enc else ["-crf","18"]
    cmd = [FFMPEG,"-y","-loop","1","-i",src,"-vf",",".join(filters),"-t",str(dur),"-c:v","libx264","-preset","fast"]+enc+["-movflags","+faststart",out]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0: raise RuntimeError(r.stderr[-800:])
    return out

# ─── PIL helpers ──────────────────────────────────────────────────────────────
def make_gradient(W, H, colors):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(img)
    c1, c2 = colors[0], colors[-1]
    mid = colors[1] if len(colors) > 2 else None
    for y in range(H):
        f = y / H
        if mid and f < 0.5:
            r = int(c1[0]+(mid[0]-c1[0])*f*2); g2 = int(c1[1]+(mid[1]-c1[1])*f*2); b = int(c1[2]+(mid[2]-c1[2])*f*2)
        elif mid:
            f2 = (f-0.5)*2; r = int(mid[0]+(c2[0]-mid[0])*f2); g2 = int(mid[1]+(c2[1]-mid[1])*f2); b = int(mid[2]+(c2[2]-mid[2])*f2)
        else:
            r = int(c1[0]+(c2[0]-c1[0])*f); g2 = int(c1[1]+(c2[1]-c1[1])*f); b = int(c1[2]+(c2[2]-c1[2])*f)
        draw.line([(0,y),(W,y)], fill=(r,g2,b))
    return img

BG_PRESETS = {
    "beach":          [(255,200,100),(255,130,50),(80,170,210)],
    "studio_white":   [(240,240,242),(225,225,228),(200,200,205)],
    "city_night":     [(8,8,25),(18,18,50),(35,35,90)],
    "forest":         [(15,55,15),(35,90,30),(25,65,20)],
    "mountain":       [(75,95,135),(95,125,175),(150,175,220)],
    "space":          [(4,4,18),(9,4,28),(18,8,45)],
    "sunset":         [(255,90,40),(220,50,80),(100,20,80)],
    "purple_gradient":[(55,15,115),(95,35,195),(35,8,75)],
    "neon_dark":      [(8,8,14),(18,8,38),(4,4,8)],
    "golden_hour":    [(230,120,10),(200,80,20),(100,30,60)],
}

def load_font(path, size):
    from PIL import ImageFont
    try: return ImageFont.truetype(path, size)
    except: return ImageFont.load_default()

FONT_BLACK = "/System/Library/Fonts/Supplemental/Arial Black.ttf"
FONT_BOLD  = "/System/Library/Fonts/Helvetica.ttc"

# ─── HTTP handler ─────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): print(f"  [{self.command}] {fmt % args}")

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin","*")
        self.send_header("Access-Control-Allow-Methods","GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers","*")

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    # ── GET ──────────────────────────────────────────────────────────────────
    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/ping":
            return self._text("OK")
        if p in ("/", "/index.html"):
            return self._serve_file(os.path.expanduser("~/Desktop/photo-animator/index.html"), "text/html")
        if p == "/files":
            return self._list_files()
        if p.startswith("/output/"):
            fname = p[8:]
            fpath = os.path.join(OUTPUT, fname)
            if os.path.isfile(fpath):
                mime = mimetypes.guess_type(fpath)[0] or "application/octet-stream"
                return self._serve_file(fpath, mime, download=True)
            return self._text("Not found", 404)
        self._text("AnimAI v2", 200)

    # ── POST ─────────────────────────────────────────────────────────────────
    def do_POST(self):
        try:
            fields, files = self._parse()
            routes = {
                "/animate":  self._animate,
                "/faceswap": self._faceswap,
                "/bgswap":   self._bgswap,
                "/retouch":  self._retouch,
                "/resize":   self._resize,
                "/grade":    self._grade,
                "/magazine": self._magazine,
            }
            handler = routes.get(self.path)
            if handler: handler(fields, files)
            else:       self._text("Unknown endpoint", 404)
        except Exception as e:
            traceback.print_exc()
            self._text(f"Error: {e}", 500)

    # ── Multipart parser ─────────────────────────────────────────────────────
    def _parse(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)
        ct     = self.headers.get("Content-Type","")
        raw    = f"Content-Type: {ct}\r\n\r\n".encode() + body
        msg    = message_from_bytes(raw, policy=HTTP)
        fields, files = {}, {}
        for part in msg.get_payload():
            cd = part.get("Content-Disposition","")
            name = fname = None
            for tok in cd.split(";"):
                tok = tok.strip()
                if tok.startswith("name="):     name  = tok[5:].strip('"')
                if tok.startswith("filename="): fname = tok[9:].strip('"')
            if not name: continue
            payload = part.get_payload(decode=True) or b""
            if fname: files[name]  = (fname, payload)
            else:     fields[name] = payload.decode("utf-8", errors="replace")
        return fields, files

    # ── Response helpers ──────────────────────────────────────────────────────
    def _text(self, body, code=200):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code); self._cors()
        self.send_header("Content-Type","text/plain")
        self.send_header("Content-Length", len(data)); self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code); self._cors()
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length", len(data)); self.end_headers()
        self.wfile.write(data)

    def _binary(self, data, mime="application/octet-stream", name="file"):
        self.send_response(200); self._cors()
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", len(data))
        self.send_header("Content-Disposition", f'attachment; filename="{name}"')
        self.end_headers(); self.wfile.write(data)

    def _serve_file(self, path, mime, download=False):
        with open(path,"rb") as f: data = f.read()
        self.send_response(200); self._cors()
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", len(data))
        if download: self.send_header("Content-Disposition",f'attachment; filename="{Path(path).name}"')
        self.end_headers(); self.wfile.write(data)

    def _list_files(self):
        out = []
        for f in sorted(Path(OUTPUT).iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[:60]:
            out.append({"name":f.name,"size":f.stat().st_size,"mtime":f.stat().st_mtime,
                        "url":f"/output/{f.name}","type":"video" if f.suffix==".mp4" else "image"})
        self._json(out)

    def _save(self, file_tuple, suffix=".jpg"):
        fname, data = file_tuple
        ext = Path(fname).suffix or suffix
        tmp = tempfile.mktemp(suffix=ext, dir="/tmp")
        with open(tmp,"wb") as f: f.write(data)
        return tmp

    def _to_jpeg(self, file_tuple):
        """Save any upload (HEIC/MOV/PNG/WEBP/JPEG) as a flat JPEG for compatibility."""
        from PIL import Image, ImageOps
        raw = self._save(file_tuple)
        out = tempfile.mktemp(suffix=".jpg", dir="/tmp")
        try:
            img = Image.open(raw).convert("RGB")
            try: img = ImageOps.exif_transpose(img)
            except: pass
            img.save(out, "JPEG", quality=95)
            return out
        except Exception as e:
            raise RuntimeError(f"Cannot decode image ({Path(file_tuple[0]).suffix}): {e}")
        finally:
            try: os.unlink(raw)
            except: pass

    def _persist(self, img_or_path, name):
        """Save PIL image or file path to OUTPUT dir."""
        from PIL import Image
        out_path = os.path.join(OUTPUT, name)
        if isinstance(img_or_path, str):
            import shutil; shutil.copy(img_or_path, out_path)
        else:
            img_or_path.save(out_path, quality=96)
        return out_path

    # ── /animate ─────────────────────────────────────────────────────────────
    def _animate(self, fields, files):
        from PIL import Image
        def g(k, d=""): v=fields.get(k,d); return v if v else d
        if "photo" not in files: return self._text("No photo",400)

        raw = self._save(files["photo"])
        # Always convert to JPEG so ffmpeg -loop 1 works regardless of
        # input format (MOV, HEIC, PNG, WEBP, video frame, etc.)
        src = tempfile.mktemp(suffix=".jpg", dir="/tmp")
        try:
            img = Image.open(raw).convert("RGB")
            # Auto-orient from EXIF
            try:
                from PIL import ImageOps
                img = ImageOps.exif_transpose(img)
            except Exception:
                pass
            img.save(src, "JPEG", quality=95)
        except Exception as e:
            return self._text(f"Cannot decode image: {e}", 400)
        finally:
            try: os.unlink(raw)
            except: pass

        out = None
        try:
            out = make_animation(src, g("style","kenburns"), g("duration","8"),
                                 g("quality","medium"), g("format","mp4"),
                                 g("grade","1"), g("vignette","1"), g("grain","0"), g("sharpen","1"))
            name = f"anim-{g('style','kb')}-{int(time.time())}.mp4"
            self._persist(out, name)
            with open(out,"rb") as f: data = f.read()
            self._binary(data,"video/mp4", name)
        finally:
            for p in [src, out]:
                if p:
                    try: os.unlink(p)
                    except: pass

    # ── /faceswap ─────────────────────────────────────────────────────────────
    def _faceswap(self, fields, files):
        import cv2
        from PIL import Image, ImageFilter
        if "source" not in files or "target" not in files:
            return self._text("Need 'source' and 'target' images", 400)
        src_path = self._to_jpeg(files["source"])
        tgt_path = self._to_jpeg(files["target"])
        try:
            app, swapper = get_face_tools()
            src_img = cv2.imread(src_path)
            tgt_img = cv2.imread(tgt_path)
            if src_img is None or tgt_img is None: return self._text("Could not read images",400)
            src_faces = app.get(src_img)
            tgt_faces = app.get(tgt_img)
            if not src_faces: return self._text("No face found in source image",400)
            if not tgt_faces: return self._text("No face found in target image",400)
            result = tgt_img.copy()
            for face in tgt_faces:
                result = swapper.get(result, face, src_faces[0], paste_back=True)
            # Sharpen face region
            pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            pil = pil.filter(ImageFilter.UnsharpMask(radius=1.5, percent=130, threshold=3))
            name = f"faceswap-{int(time.time())}.jpg"
            out_path = self._persist(pil, name)
            with open(out_path,"rb") as f: data = f.read()
            self._binary(data,"image/jpeg", name)
        finally:
            for p in [src_path, tgt_path]:
                try: os.unlink(p)
                except: pass

    # ── /bgswap ──────────────────────────────────────────────────────────────
    def _bgswap(self, fields, files):
        from PIL import Image
        from rembg import remove
        if "photo" not in files: return self._text("Need photo",400)
        photo_path = self._save(files["photo"])
        preset = fields.get("background","beach")
        try:
            with open(photo_path,"rb") as f: fg_data = remove(f.read())
            fg = Image.open(io.BytesIO(fg_data)).convert("RGBA")
            W, H = fg.size
            colors = BG_PRESETS.get(preset, BG_PRESETS["beach"])
            bg = make_gradient(W, H, colors).convert("RGBA")
            bg.paste(fg, (0,0), fg)
            result = bg.convert("RGB")
            name = f"bgswap-{preset}-{int(time.time())}.jpg"
            out_path = self._persist(result, name)
            with open(out_path,"rb") as f: data = f.read()
            self._binary(data,"image/jpeg", name)
        finally:
            try: os.unlink(photo_path)
            except: pass

    # ── /retouch ─────────────────────────────────────────────────────────────
    def _retouch(self, fields, files):
        import numpy as np
        from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
        if "photo" not in files: return self._text("Need photo",400)
        photo_path = self._save(files["photo"])
        intensity = fields.get("intensity","medium")
        blur = {"light":2,"medium":3,"heavy":5}.get(intensity,3)
        try:
            img = Image.open(photo_path).convert("RGB")
            W, H = img.size
            # 1. Skin smooth
            low = img.filter(ImageFilter.GaussianBlur(blur))
            hi  = np.array(img, dtype=np.float32) - np.array(low, dtype=np.float32) + 128
            arr = np.array(low, dtype=np.float32)*0.82 + hi*0.18
            img = Image.fromarray(np.clip(arr,0,255).astype(np.uint8))
            # 2. Color grade
            img = ImageEnhance.Contrast(img).enhance(1.12)
            img = ImageEnhance.Color(img).enhance(1.18)
            img = ImageEnhance.Brightness(img).enhance(1.05)
            # 3. Clarity
            img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=5))
            # 4. Warm
            arr = np.array(img, dtype=np.float32)
            arr[:,:,0] = np.clip(arr[:,:,0]*1.04, 0, 255)
            arr[:,:,2] = np.clip(arr[:,:,2]*0.97, 0, 255)
            img = Image.fromarray(arr.astype(np.uint8))
            # 5. Vignette
            mask = Image.new("L",(W,H),255)
            vd = ImageDraw.Draw(mask)
            for i in range(min(W,H)//4):
                a = 255 - int(100*(i/(min(W,H)//4)))
                vd.ellipse([i,i*H//W,W-i,H-i*H//W], fill=a)
            dark = Image.new("RGB",(W,H),(0,0,0))
            img  = Image.composite(img, dark, mask)
            # 6. Final sharpen
            img = img.filter(ImageFilter.UnsharpMask(radius=0.8, percent=80, threshold=2))
            name = f"retouch-{intensity}-{int(time.time())}.jpg"
            out_path = self._persist(img, name)
            with open(out_path,"rb") as f: data = f.read()
            self._binary(data,"image/jpeg", name)
        finally:
            try: os.unlink(photo_path)
            except: pass

    # ── /resize ──────────────────────────────────────────────────────────────
    def _resize(self, fields, files):
        from PIL import Image
        if "photo" not in files: return self._text("Need photo",400)
        photo_path = self._save(files["photo"])
        try:
            W = int(fields.get("width","1920") or 1920)
            H = int(fields.get("height","1080") or 1080)
            fmt   = fields.get("format","jpg").lower().replace("jpeg","jpg")
            mode  = fields.get("mode","fit")
            qual  = int(fields.get("quality","95") or 95)
            img   = Image.open(photo_path).convert("RGB")
            if mode == "fill":
                ir = img.width/img.height; tr = W/H
                nw = int(H*ir) if ir>tr else W
                nh = H         if ir>tr else int(W/ir)
                img = img.resize((nw,nh), Image.LANCZOS)
                img = img.crop(((nw-W)//2,(nh-H)//2,(nw-W)//2+W,(nh-H)//2+H))
            elif mode == "stretch":
                img = img.resize((W,H), Image.LANCZOS)
            else:
                img.thumbnail((W,H), Image.LANCZOS)
            ext_map = {"jpg":".jpg","png":".png","webp":".webp"}
            ext  = ext_map.get(fmt,".jpg")
            name = f"resize-{img.width}x{img.height}-{int(time.time())}{ext}"
            out_path = os.path.join(OUTPUT, name)
            pil_fmt  = {"jpg":"JPEG","png":"PNG","webp":"WEBP"}.get(fmt,"JPEG")
            if pil_fmt == "PNG": img.save(out_path, format="PNG", optimize=True)
            else:                img.save(out_path, format=pil_fmt, quality=qual, optimize=True)
            with open(out_path,"rb") as f: data = f.read()
            mime = {"jpg":"image/jpeg","png":"image/png","webp":"image/webp"}.get(fmt,"image/jpeg")
            self._binary(data, mime, name)
        finally:
            try: os.unlink(photo_path)
            except: pass

    # ── /grade ───────────────────────────────────────────────────────────────
    def _grade(self, fields, files):
        import numpy as np
        from PIL import Image, ImageEnhance, ImageDraw
        if "photo" not in files: return self._text("Need photo",400)
        photo_path = self._save(files["photo"])
        preset = fields.get("preset","cinematic")
        GRADES = {
            "cinematic":    dict(cn=1.15,co=0.85,r=1.02,g=1.00,b=0.95,br=0.95),
            "warm":         dict(cn=1.08,co=1.10,r=1.08,g=1.02,b=0.88,br=1.05),
            "cool":         dict(cn=1.10,co=0.90,r=0.92,g=0.97,b=1.10,br=1.00),
            "vintage":      dict(cn=0.95,co=0.70,r=1.05,g=1.00,b=0.85,br=1.02),
            "golden_hour":  dict(cn=1.12,co=1.15,r=1.12,g=1.05,b=0.78,br=1.08),
            "night":        dict(cn=1.20,co=0.80,r=0.88,g=0.92,b=1.18,br=0.88),
            "moody":        dict(cn=1.25,co=0.75,r=0.95,g=0.92,b=0.98,br=0.85),
            "matte":        dict(cn=0.88,co=0.82,r=1.00,g=0.98,b=0.95,br=1.12),
            "hdr":          dict(cn=1.30,co=1.20,r=1.05,g=1.02,b=1.00,br=0.98),
            "pastel":       dict(cn=0.85,co=0.60,r=1.03,g=1.02,b=1.05,br=1.15),
        }
        try:
            p   = GRADES.get(preset, GRADES["cinematic"])
            img = Image.open(photo_path).convert("RGB")
            arr = np.array(img, dtype=np.float32)
            arr[:,:,0] = np.clip(arr[:,:,0]*p["r"], 0, 255)
            arr[:,:,1] = np.clip(arr[:,:,1]*p["g"], 0, 255)
            arr[:,:,2] = np.clip(arr[:,:,2]*p["b"], 0, 255)
            img = Image.fromarray(arr.astype(np.uint8))
            img = ImageEnhance.Contrast(img).enhance(p["cn"])
            img = ImageEnhance.Color(img).enhance(p["co"])
            img = ImageEnhance.Brightness(img).enhance(p["br"])
            if preset in ("cinematic","moody","night","vintage"):
                W2,H2 = img.size
                mask = Image.new("L",(W2,H2),255)
                md = ImageDraw.Draw(mask)
                for i in range(min(W2,H2)//4):
                    md.ellipse([i,i*H2//W2,W2-i,H2-i*H2//W2], fill=255-int(100*i/(min(W2,H2)//4)))
                img = Image.composite(img, Image.new("RGB",(W2,H2),(0,0,0)), mask)
            name = f"grade-{preset}-{int(time.time())}.jpg"
            out_path = self._persist(img, name)
            with open(out_path,"rb") as f: data = f.read()
            self._binary(data,"image/jpeg", name)
        finally:
            try: os.unlink(photo_path)
            except: pass

    # ── /magazine ─────────────────────────────────────────────────────────────
    def _magazine(self, fields, files):
        from PIL import Image, ImageDraw, ImageFont
        if "photo" not in files: return self._text("Need photo",400)
        photo_path = self._save(files["photo"])
        template   = fields.get("template","vogue")
        W, H = 1200, 1600
        try:
            photo = Image.open(photo_path).convert("RGB")
            ratio = max(W/photo.width, H/photo.height)
            nw, nh = int(photo.width*ratio), int(photo.height*ratio)
            photo  = photo.resize((nw,nh), Image.LANCZOS)
            left   = (nw-W)//2
            top    = max(0, (nh-H)//4)
            cover  = photo.crop((left,top,left+W,top+H))
            draw   = ImageDraw.Draw(cover)

            def fnt(size, bold=True):
                return load_font(FONT_BLACK if bold else FONT_BOLD, size)

            def bottom_grad(img, height=280, alpha=210):
                from PIL import Image as _I
                g = _I.new("RGBA",(W,height),(0,0,0,0))
                gd = ImageDraw.Draw(g)
                for y in range(height): gd.line([(0,y),(W,y)], fill=(0,0,0,int(alpha*y/height)))
                c = img.convert("RGBA"); c.paste(g,(0,H-height),g); return c.convert("RGB")

            def top_overlay(img, h=180, alpha=140):
                from PIL import Image as _I
                ov = _I.new("RGBA",(W,h),(0,0,0,alpha))
                c  = img.convert("RGBA"); c.paste(ov,(0,0),ov); return c.convert("RGB")

            if template == "vogue":
                cover = top_overlay(cover, 190, 100)
                draw  = ImageDraw.Draw(cover)
                draw.text((-10,-12), "VOGUE",          font=fnt(290), fill=(255,255,255))
                draw.rectangle([0,175,W,181],           fill=(180,140,60))
                draw.text((14,186), "FEBRUARY 2026  •  THE POWER ISSUE", font=fnt(27,False), fill=(255,255,255))
                cover = bottom_grad(cover, 260, 200)
                draw  = ImageDraw.Draw(cover)
                draw.text((14,H-230), "THE FUTURE IS NOW",                        font=fnt(42), fill=(255,255,255))
                draw.text((14,H-180), "Exclusives · Beauty · Power Players",      font=fnt(24,False), fill=(210,210,210))
                draw.text((14,H-140), "Rihanna · Zendaya · Blackpink",            font=fnt(24,False), fill=(180,180,180))
                draw.text((14,H-100), "Spring Collections · Oscars Preview",      font=fnt(22,False), fill=(160,160,160))
                draw.rectangle([0,H-55,W,H],            fill=(0,0,0))
                draw.text((14,H-40),  "ANIMAI STUDIO · LOCAL AI EDITION",         font=fnt(20,False), fill=(120,120,120))

            elif template == "si_swimsuit":
                cover = top_overlay(cover, 165, 150)
                draw  = ImageDraw.Draw(cover)
                draw.text((8,-14), "Sports Illustrated", font=fnt(118), fill=(210,25,25))
                draw.rectangle([0,128,W,135],          fill=(255,255,255))
                draw.text((14,140), "SWIMSUIT EDITION 2026",  font=fnt(50), fill=(255,255,255))
                cover = bottom_grad(cover, 280, 200)
                draw  = ImageDraw.Draw(cover)
                draw.text((14,H-255), "50 YEARS OF ICONIC COVERS",               font=fnt(28,False), fill=(210,25,25))
                draw.text((14,H-215), "The Models · The Locations · The Legacy",  font=fnt(26,False), fill=(255,255,255))
                draw.text((14,H-170), "Paradise Found · Body Confidence · Style", font=fnt(24,False), fill=(200,200,200))
                draw.text((14,H-120), "Exclusive Behind-the-Lens Access",         font=fnt(24,False), fill=(170,170,170))
                draw.rectangle([0,H-60,W,H],           fill=(20,20,20))
                draw.text((14,H-44),  "ANIMAI LOCAL AI STUDIO",                   font=fnt(20,False), fill=(130,130,130))

            elif template == "maxim":
                cover = top_overlay(cover, 160, 130)
                draw  = ImageDraw.Draw(cover)
                draw.text((-5,-22), "MAXIM",           font=fnt(240), fill=(210,100,20))
                draw.rectangle([0,148,W,155],          fill=(210,100,20))
                draw.rounded_rectangle([14,158,360,198], radius=5, fill=(160,70,10))
                draw.text((22,164), "AMERICAN MAVERICK  ★  2026",                 font=fnt(22), fill=(255,220,100))
                cover = bottom_grad(cover, 300, 210)
                draw  = ImageDraw.Draw(cover)
                draw.text((14,H-275), "HOT 100 · POWER LIST · COVER STORY",      font=fnt(26,False), fill=(210,100,20))
                draw.text((14,H-235), "Inside Track · The Best of 2026",          font=fnt(26,False), fill=(255,255,255))
                draw.text((14,H-190), "Style Files · Gear · Exclusive Interview", font=fnt(24,False), fill=(200,200,200))
                draw.text((14,H-145), "Behind the Scenes · Power Players",        font=fnt(22,False), fill=(170,170,170))
                draw.rectangle([0,H-65,W,H],           fill=(0,0,0))
                draw.text((14,H-47),  "ANIMAI.LOCAL · AI CREATION STUDIO",       font=fnt(20,False), fill=(130,130,130))

            elif template == "rolling_stone":
                cover = top_overlay(cover, 145, 160)
                draw  = ImageDraw.Draw(cover)
                draw.text((8,6), "Rolling Stone",      font=fnt(108), fill=(220,25,25))
                draw.rectangle([0,126,W,133],          fill=(220,25,25))
                draw.text((14,138), "THE MUSIC · THE CULTURE · THE POLITICS",     font=fnt(26,False), fill=(220,220,220))
                cover = bottom_grad(cover, 300, 200)
                draw  = ImageDraw.Draw(cover)
                draw.text((14,H-280), "COVER STORY",                              font=fnt(28,False), fill=(220,25,25))
                draw.text((14,H-240), "The Artist of Our Time",                   font=fnt(58), fill=(255,255,255))
                draw.text((14,H-175), "PLUS: The New Sound of 2026",              font=fnt(26,False), fill=(200,200,200))
                draw.text((14,H-135), "Festival Preview · Industry Power Players",font=fnt(24,False), fill=(170,170,170))
                draw.rectangle([0,H-65,W,H],           fill=(20,20,20))
                draw.text((14,H-47),  "ANIMAI STUDIO SPECIAL EDITION",            font=fnt(20,False), fill=(120,120,120))

            elif template == "forbes":
                cover = top_overlay(cover, 195, 145)
                draw  = ImageDraw.Draw(cover)
                draw.text((8,8), "Forbes",             font=fnt(148), fill=(220,25,25))
                draw.rectangle([0,178,W,186],          fill=(220,25,25))
                draw.text((14,192), "THE WORLD'S MOST POWERFUL  •  2026",         font=fnt(26,False), fill=(255,255,255))
                cover = bottom_grad(cover, 330, 210)
                draw  = ImageDraw.Draw(cover)
                draw.text((14,H-305), "#1 ON THE POWER LIST",                     font=fnt(26,False), fill=(220,25,25))
                draw.text((14,H-265), "The Billionaire Next Door",                 font=fnt(48), fill=(255,255,255))
                draw.text((14,H-205), "Building a $1B Brand From Scratch",         font=fnt(28,False), fill=(200,200,200))
                draw.text((14,H-165), "30 Under 30  ·  AI & Wealth  ·  Real Estate",font=fnt(24,False), fill=(175,175,175))
                draw.text((14,H-120), "New Economy  ·  Crypto Comeback  ·  Tech",  font=fnt(22,False), fill=(150,150,150))
                draw.rectangle([0,H-65,W,H],           fill=(20,20,20))
                draw.text((14,H-47),  "ANIMAI STUDIO  ·  LOCAL AI SUITE",          font=fnt(20,False), fill=(120,120,120))

            name = f"magazine-{template}-{int(time.time())}.jpg"
            out_path = self._persist(cover, name)
            with open(out_path,"rb") as f: data = f.read()
            self._binary(data,"image/jpeg", name)
        finally:
            try: os.unlink(photo_path)
            except: pass

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════╗
║   ✨ AnimAI Server v2 — All Tools       ║
║   http://localhost:7474                  ║
║   Output: ~/Desktop/photo-animator/output║
╚══════════════════════════════════════════╝""")
    HTTPServer(("", 7474), Handler).serve_forever()
