"""
Oil Pastel — transforma uma foto numa pintura de giz pastel oleoso,
pintada ao vivo em camadas (pinceladas grossas -> finas).

Uso:
    pip install pillow numpy
    python pastel.py minha_foto.jpg
    python pastel.py minha_foto.jpg --save pintura.png --gif processo.gif
    python pastel.py minha_foto.jpg --no-window --save pintura.png
    python pastel.py girassois.jpg --loop --duration 12   # 12 s por pintura, em loop
"""
import argparse
import math
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ---------------------------------------------------------------- ajustes
MAX_H = 560                 # altura máxima da janela/pintura
PAPER = (236, 228, 212)     # cor do papel
PAPER_TEXTURE = 0.10        # quanto a granulação do papel aparece
TOOTH = 0.55                # "dente" do papel: >0 faz o giz falhar nas partes baixas
STROKE_SCALE = 1.0          # multiplica a largura de todas as pinceladas
COLOR_JITTER = 10           # variação de cor por pincelada (0-255)
SATURATION = 1.2            # pastel costuma ser mais saturado que a foto
BATCH = 120                 # pinceladas por quadro (só no modo --no-window/GIF)
DURATION = 12.0             # segundos que a pintura leva para ficar pronta na janela
LOOP_PAUSE_MS = 1000        # com --loop: quanto tempo a pintura pronta fica na tela

# cada camada: largura (fração da altura), comprimento máximo (em passos),
# densidade (pinceladas por área de pincel) e limiar de erro
LAYERS = [
    {"w": 0.045, "len": 5,  "density": 2.5, "thresh": 0.00},
    {"w": 0.024, "len": 7,  "density": 1.6, "thresh": 0.10},
    {"w": 0.013, "len": 8,  "density": 1.2, "thresh": 0.25},
    {"w": 0.007, "len": 9,  "density": 0.8, "thresh": 0.45},
    {"w": 0.004, "len": 6,  "density": 0.7, "thresh": 0.50},   # olhos, boca, contornos
]


# ---------------------------------------------------------------- utilidades
def _box(a, r, axis):
    """Média móvel (box blur) com somas acumuladas, bordas replicadas."""
    if r < 1:
        return a
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float64)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return ((hi - lo) / (2 * r + 1)).astype(np.float32)


def gblur(a, sigma):
    """Gaussian blur aproximado (3 box blurs) num array float 2D — só numpy."""
    if sigma < 0.4:
        return a.astype(np.float32)
    r = max(1, int(round((math.sqrt(4 * sigma * sigma + 1) - 1) / 2)))
    for _ in range(3):
        a = _box(_box(a, r, 0), r, 1)
    return a


def make_grain(W, H, rng):
    """Granulação de papel: ruído em várias escalas, normalizado para 0..1."""
    g = np.zeros((H, W), np.float32)
    for s, amp in ((0.6, 0.5), (1.5, 0.3), (4.0, 0.2)):
        g += amp * gblur(rng.random((H, W)).astype(np.float32), s)
    g -= g.min()
    return g / (g.max() + 1e-6)


def boost(rgb, sat):
    gray = rgb.mean()
    return np.clip(gray + (rgb - gray) * sat, 0, 255)


# ---------------------------------------------------------------- pintor
class Pastel:
    def __init__(self, W, H, photo, seed=None):
        self.W, self.H = W, H
        self.rng = np.random.default_rng(seed)
        self.grain = make_grain(W, H, self.rng)

        paper = np.array(PAPER, np.float32)[None, None, :]
        paper = paper * (1 + PAPER_TEXTURE * (self.grain[..., None] - 0.5) * 2)
        self.paper = np.clip(paper, 0, 255).astype(np.float32)
        self.canvas = self.paper.copy()
        self.photo = photo
        self.strokes = self.build(photo)
        self.flat = [s for layer in self.strokes for s in layer]
        self.painted = 0

    # planeja todas as pinceladas antes de pintar
    def build(self, rgb_img):
        W, H, rng = self.W, self.H, self.rng
        rgbf = np.asarray(rgb_img, np.float32) / 255.0
        lum = rgbf @ np.array([0.299, 0.587, 0.114], np.float32)

        # direção das pinceladas: tensor de estrutura (segue as bordas)
        g = gblur(lum, 1.0)
        gy, gx = np.gradient(g)
        parts = []
        prev_col = None
        for li, ly in enumerate(LAYERS):
            w = max(2.0, H * ly["w"] * STROKE_SCALE)
            col_k = np.asarray(rgb_img.filter(ImageFilter.GaussianBlur(w * 0.30)), np.float32)

            # onde pintar: tudo na 1ª camada, depois só onde há detalhe novo
            if prev_col is None:
                err = np.ones((H, W), np.float32)
            else:
                err = np.abs(col_k - prev_col).mean(axis=2)
                err = np.clip(err / (np.percentile(err, 92) + 1e-6), 0, 1)
            prev_col = col_k

            s = max(1.0, w * 0.9)
            jxx, jyy, jxy = gblur(gx * gx, s), gblur(gy * gy, s), gblur(gx * gy, s)
            theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + math.pi / 2
            coher = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-9)
            # onde não há borda clara, usa um ângulo "da mão" com leve variação
            hand = math.radians(-35) + gblur(rng.normal(0, 1, (H, W)).astype(np.float32), w * 3) * 6
            theta = np.where(coher > 0.25, theta, hand)
            dx, dy = np.cos(theta), np.sin(theta)

            if li == 0:
                # 1ª camada: grade com leve variação, para não sobrar papel em branco
                gap = w * 0.6
                gx0, gy0 = np.meshgrid(np.arange(0, W, gap), np.arange(0, H, gap))
                n = gx0.size
                xs = np.clip(gx0.ravel() + rng.uniform(-0.4, 0.4, n) * gap, 0, W - 1)
                ys = np.clip(gy0.ravel() + rng.uniform(-0.4, 0.4, n) * gap, 0, H - 1)
            else:
                n = int(ly["density"] * W * H / (w * w))
                xs = rng.uniform(0, W - 1, n)
                ys = rng.uniform(0, H - 1, n)
            xi, yi = xs.astype(int), ys.astype(int)
            keep = err[yi, xi] > ly["thresh"] + rng.random(n) * 0.35 * (li > 0)
            xs, ys = xs[keep], ys[keep]

            layer = []
            step = w * 0.55
            for x0, y0 in zip(xs, ys):
                ix, iy = int(x0), int(y0)
                c = boost(col_k[iy, ix], SATURATION) + rng.normal(0, COLOR_JITTER, 3)
                c = tuple(int(v) for v in np.clip(c, 0, 255))
                pts = [(x0, y0)]
                x, y = x0, y0
                vx, vy = dx[iy, ix], dy[iy, ix]
                L = rng.integers(2, ly["len"] + 1)
                for _ in range(L):
                    ix, iy = int(x), int(y)
                    nx, ny = dx[iy, ix], dy[iy, ix]
                    if nx * vx + ny * vy < 0:      # mantém o sentido
                        nx, ny = -nx, -ny
                    vx, vy = 0.6 * vx + 0.4 * nx, 0.6 * vy + 0.4 * ny
                    nrm = math.hypot(vx, vy) or 1
                    vx, vy = vx / nrm, vy / nrm
                    x, y = x + vx * step, y + vy * step
                    if not (0 <= x < W and 0 <= y < H):
                        break
                    # para quando a cor embaixo muda demais (respeita bordas)
                    if np.abs(col_k[int(y), int(x)] - col_k[int(y0), int(x0)]).mean() > 28:
                        break
                    pts.append((x, y))
                if len(pts) == 1:
                    pts.append((x0 + vx * step * 0.5, y0 + vy * step * 0.5))
                ww = w * rng.uniform(0.8, 1.15)
                layer.append((pts, c, ww))

            # pinceladas maiores/mais claras primeiro deixa mais natural
            rng.shuffle(layer)
            parts.append(layer)
        return parts

    def paint_batch(self, batch):
        """Desenha um lote de pinceladas com textura de giz e compõe no canvas."""
        W, H = self.W, self.H
        col = Image.new("RGB", (W, H))
        mask = Image.new("L", (W, H), 0)
        dc, dm = ImageDraw.Draw(col), ImageDraw.Draw(mask)
        for pts, c, w in batch:
            wi = max(1, int(w))
            dc.line(pts, fill=c, width=wi, joint="curve")
            dm.line(pts, fill=255, width=wi, joint="curve")
            r = wi / 2
            for (px, py) in (pts[0], pts[-1]):            # pontas arredondadas
                dc.ellipse((px - r, py - r, px + r, py + r), fill=c)
                dm.ellipse((px - r, py - r, px + r, py + r), fill=255)
        a = np.asarray(mask, np.float32) / 255.0
        # o giz pega nos "picos" do papel e falha nos "vales"
        tooth = np.clip((self.grain - (TOOTH - 0.5) * 0.6 - 0.2) * 2.2, 0, 1)
        a = a * (0.55 + 0.45 * tooth)
        a = a[..., None]
        self.canvas = self.canvas * (1 - a) + np.asarray(col, np.float32) * a

    def reset(self):
        """Volta ao papel em branco (as pinceladas planejadas são mantidas)."""
        self.canvas = self.paper.copy()
        self.painted = 0

    def paint_upto(self, n):
        """Pinta as pinceladas que faltam até a n-ésima (na ordem das camadas)."""
        if n > self.painted:
            self.paint_batch(self.flat[self.painted:n])
            self.painted = n

    def frames(self, batch=BATCH):
        """Gera a pintura quadro a quadro (cada quadro = um lote de pinceladas)."""
        for layer in self.strokes:
            for i in range(0, len(layer), batch):
                self.paint_batch(layer[i:i + batch])
                yield self.image()

    def image(self):
        return Image.fromarray(np.clip(self.canvas, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------- execução
def load(path):
    img = Image.open(path).convert("RGB")
    s = MAX_H / img.height          # fotos pequenas também são ampliadas
    return img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser(description="Foto -> pintura em giz pastel")
    ap.add_argument("image")
    ap.add_argument("--save", help="salva a pintura final (png/jpg)")
    ap.add_argument("--gif", help="salva o processo como GIF")
    ap.add_argument("--no-window", action="store_true", help="não abre janela")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--batch", type=int, default=BATCH,
                    help="pinceladas por quadro no modo --no-window")
    ap.add_argument("--duration", type=float, default=DURATION,
                    help="segundos que a pintura leva na janela")
    ap.add_argument("--loop", action="store_true",
                    help="recomeça a pintura ao terminar")
    args = ap.parse_args()

    photo = load(args.image)
    W, H = photo.size
    print(f"Planejando pinceladas para {W}x{H}...")
    p = Pastel(W, H, photo, args.seed)
    print("Pinceladas por camada:", [len(l) for l in p.strokes])

    gif_frames = []

    def keep(frame, k):
        if args.gif and k % 3 == 0:
            gif_frames.append(frame.resize((W // 2, H // 2)))

    if args.no_window:
        for k, f in enumerate(p.frames(args.batch)):
            keep(f, k)
        finish(p, args, gif_frames)
        return

    import tkinter as tk
    from PIL import ImageTk

    root = tk.Tk()
    root.title("Oil Pastel")
    # abre onde a janela estava da última vez (arraste para onde quiser);
    # na primeira vez, abre no centro da tela
    from pathlib import Path
    pos_file = Path(__file__).with_name(".janela_pos")
    try:
        pos = pos_file.read_text().strip()
    except OSError:
        pos = ""
    if not pos:
        x = max(0, (root.winfo_screenwidth() - p.W) // 2)
        y = max(0, (root.winfo_screenheight() - p.H) // 2)
        pos = f"+{x}+{y}"
    root.geometry(pos)
    last = {"pos": pos}

    def remember(_e=None):
        g = root.geometry()                 # "LxA+X+Y"
        last["pos"] = g[g.index("+"):] if "+" in g else last["pos"]

    root.bind("<Configure>", remember)
    label = tk.Label(root, bg="black")
    label.pack()
    state = {"k": 0, "t0": 0.0, "done": False}
    total = len(p.flat)

    def start():
        p.reset()
        state["t0"] = time.perf_counter()
        tick()

    def tick():
        # o ritmo segue o relógio: termina em args.duration seja qual for a máquina
        t = (time.perf_counter() - state["t0"]) / max(args.duration, 0.1)
        p.paint_upto(int(total * min(1.0, t)))
        f = p.image()
        if not state["done"]:
            keep(f, state["k"])
            state["k"] += 1
        tkimg = ImageTk.PhotoImage(f)
        label.configure(image=tkimg)
        label.image = tkimg
        if p.painted < total:
            root.after(16, tick)
            return
        if not state["done"]:                      # salva só na 1ª vez
            state["done"] = True
            finish(p, args, gif_frames)
        if args.loop:
            root.after(LOOP_PAUSE_MS, start)

    root.after(200, start)
    try:
        root.mainloop()
    finally:
        try:
            pos_file.write_text(last["pos"])
        except OSError:
            pass


def finish(p, args, gif_frames):
    final = p.image()
    if args.save:
        final.save(args.save)
        print("Salvo:", args.save)
    if args.gif:
        small = final.resize(gif_frames[0].size) if gif_frames else final
        frames = gif_frames + [small] * 25
        frames[0].save(args.gif, save_all=True, append_images=frames[1:],
                       duration=60, loop=0, optimize=True)
        print("GIF salvo:", args.gif)


if __name__ == "__main__":
    main()
