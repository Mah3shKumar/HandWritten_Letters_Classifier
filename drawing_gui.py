import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw
import joblib
import os
from collections import Counter


# ============================================
# UPDATE THIS PATH TO YOUR LOCAL EMNIST FOLDER
# ============================================
DATA_PATH = r"F:\IDE\Python\PyP\emnist_source_files"

ALPHABET  = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')


print("Loading model and data...")
model  = joblib.load(os.path.join(DATA_PATH, 'best_knn_model.joblib'))
subset = np.load(os.path.join(DATA_PATH, 'emnist_subset.npz'))
y_tr   = subset['y_tr']
print("  Model loaded successfully.")


def preprocess(pil_image):
    img = pil_image.convert('L')
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = arr / 255.0
    coords = np.argwhere(arr > 0.1)

    if coords.shape[0] == 0:
        return arr

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    cropped = arr[y_min:y_max+1, x_min:x_max+1]

    h, w = cropped.shape
    if h > w * 3:
        pad = 6  
    else:
        pad = 3 


    cropped_pil  = Image.fromarray((cropped * 255).astype(np.uint8))
    target_size  = 28 - (pad * 2)
    cropped_pil  = cropped_pil.resize(
        (target_size, target_size), Image.LANCZOS
    )

    new_img = Image.new('L', (28, 28), 0)
    new_img.paste(cropped_pil, (pad, pad))

    arr = np.array(new_img, dtype=np.float32) / 255.0
    return arr



class DrawingApp:

    CANVAS_SIZE = 280
    BRUSH_SIZE  = 20

    def __init__(self, root):
        self.root = root
        self.root.title("Data Craft — Handwritten Letter Classifier")
        self.root.geometry("440x680")
        self.root.configure(bg='#1a1a2e')
        self._reset_pil()
        self._build_ui()
        self.canvas.bind('<B1-Motion>',     self.draw)
        self.canvas.bind('<ButtonPress-1>', self.draw)

    def _reset_pil(self):
        self.pil_img  = Image.new('RGB',
                                  (self.CANVAS_SIZE, self.CANVAS_SIZE),
                                  'black')
        self.pil_draw = ImageDraw.Draw(self.pil_img)

    def _build_ui(self):
        tk.Label(self.root,
                 text="Data Craft — Handwritten Letter Classifier",
                 bg='#1a1a2e', fg='white',
                 font=('Arial', 13, 'bold')).pack(pady=(10, 2))

        tk.Label(self.root,
                 text="Team: Data Craft  |  kNN (k=5)  |  82.12% Accuracy",
                 bg='#1a1a2e', fg='#94a3b8',
                 font=('Arial', 8)).pack()

        tk.Label(self.root,
                 text="Draw letter outline — do not fill inside",
                 bg='#1a1a2e', fg='#4cc9f0',
                 font=('Arial', 9)).pack(pady=(2, 8))

        self.canvas = tk.Canvas(self.root,
                                width=self.CANVAS_SIZE,
                                height=self.CANVAS_SIZE,
                                bg='black',
                                cursor='crosshair',
                                highlightthickness=2,
                                highlightbackground='#4cc9f0')
        self.canvas.pack()

        tk.Label(self.root,
                 text="28×28 Preview (what model sees)",
                 bg='#1a1a2e', fg='#94a3b8',
                 font=('Arial', 8)).pack(pady=(8, 2))

        self.preview = tk.Canvas(self.root,
                                 width=84, height=84,
                                 bg='black',
                                 highlightthickness=1,
                                 highlightbackground='#94a3b8')
        self.preview.pack()

        self.result = tk.Label(self.root,
                               text="?",
                               font=('Arial', 48, 'bold'),
                               bg='#1a1a2e', fg='#4cc9f0')
        self.result.pack(pady=(8, 0))

        self.conf = tk.Label(self.root,
                             text="Draw a letter then click PREDICT",
                             bg='#1a1a2e', fg='#94a3b8',
                             font=('Arial', 9))
        self.conf.pack()

        self.top3 = tk.Label(self.root,
                             text="—",
                             bg='#1a1a2e', fg='white',
                             font=('Courier', 10, 'bold'))
        self.top3.pack(pady=(2, 8))

        tk.Button(self.root,
                  text="  PREDICT  ",
                  font=('Arial', 11, 'bold'),
                  bg='#4361ee', fg='white',
                  relief='flat', padx=14, pady=8,
                  cursor='hand2',
                  command=self.predict).pack(pady=(0, 4))

        tk.Button(self.root,
                  text="  CLEAR  ",
                  font=('Arial', 11, 'bold'),
                  bg='#e63946', fg='white',
                  relief='flat', padx=14, pady=8,
                  cursor='hand2',
                  command=self.clear).pack()

    def draw(self, event):
        x, y = event.x, event.y
        r    = self.BRUSH_SIZE

        self.canvas.create_oval(x-r, y-r, x+r, y+r,
                                fill='white', outline='white')
        self.pil_draw.ellipse([x-r, y-r, x+r, y+r],
                              fill='white', outline='white')

    def predict(self):
        arr = preprocess(self.pil_img)
        self.show_preview(arr)
        img_flat = arr.flatten().reshape(1, -1)
        pred   = model.predict(img_flat)[0]
        letter = ALPHABET[pred]
        distances, indices = model.kneighbors(img_flat)
        neighbor_labels    = y_tr[indices[0]]

        votes      = Counter(neighbor_labels.tolist())
        total      = sum(votes.values())
        confidence = votes[pred] / total * 100

        top3_list = votes.most_common(3)
        top3_str  = "     ".join([
            f"{ALPHABET[lbl]}  {cnt/total*100:.0f}%"
            for lbl, cnt in top3_list
        ])

        self.result.config(text=letter)
        self.conf.config(
            text=f"Confidence: {confidence:.1f}%",
            fg='#4ade80' if confidence >= 60 else '#fb923c'
        )
        self.top3.config(text=top3_str)

        print(f"  Predicted : {letter}")
        print(f"  Confidence: {confidence:.1f}%")
        print(f"  Top 3     : {top3_str}")

    def show_preview(self, arr):
        self.preview.delete('all')

        for i in range(28):
            for j in range(28):
                val   = int(arr[i, j] * 255)
                color = f'#{val:02x}{val:02x}{val:02x}'
                self.preview.create_rectangle(
                    j*3, i*3, j*3+3, i*3+3,
                    fill=color, outline=''
                )

    def clear(self):
        self.canvas.delete('all')
        self.preview.delete('all')
        self._reset_pil()

        self.result.config(text='?', fg='#4cc9f0')
        self.conf.config(
            text='Draw a letter then click PREDICT',
            fg='#94a3b8'
        )
        self.top3.config(text='—')

        print("  Canvas cleared — ready for new drawing.")


print("\nLaunching Drawing GUI...")
print("  1. Draw a letter outline in the black box")
print("  2. Do NOT fill inside the letter")
print("  3. Click PREDICT to classify")
print("  4. Click CLEAR to draw again")
print()

root = tk.Tk()
app  = DrawingApp(root)
root.mainloop()
