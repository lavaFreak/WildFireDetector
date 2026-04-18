from __future__ import annotations

from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ModuleNotFoundError:
    tk = None
    filedialog = None
    messagebox = None
    ttk = None

from src.inference import DEFAULT_THRESHOLD, collect_image_paths, ensemble_predict, load_classifier, predict_with_classifier


class WildfireClassifierApp:
    def __init__(self, root: "tk.Tk") -> None:
        self.root = root
        self.root.title("WildFireDetector")
        self.root.geometry("980x620")

        self.selected_paths: list[Path] = []

        controls = ttk.Frame(root, padding=12)
        controls.pack(fill="x")

        ttk.Button(controls, text="Choose Images", command=self.choose_images).grid(row=0, column=0, padx=6, pady=6)
        ttk.Button(controls, text="Choose Folder", command=self.choose_folder).grid(row=0, column=1, padx=6, pady=6)

        ttk.Label(controls, text="Model").grid(row=0, column=2, padx=(18, 6), pady=6)
        self.model_var = tk.StringVar(value="ensemble")
        ttk.Combobox(
            controls,
            textvariable=self.model_var,
            values=("ensemble", "cnn_64x64", "cnn_16x16"),
            state="readonly",
            width=14,
        ).grid(row=0, column=3, padx=6, pady=6)

        ttk.Label(controls, text="Threshold").grid(row=0, column=4, padx=(18, 6), pady=6)
        self.threshold_var = tk.StringVar(value=str(DEFAULT_THRESHOLD))
        ttk.Entry(controls, textvariable=self.threshold_var, width=8).grid(row=0, column=5, padx=6, pady=6)

        self.tta_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Flip TTA", variable=self.tta_var).grid(row=0, column=6, padx=12, pady=6)
        ttk.Button(controls, text="Classify", command=self.classify).grid(row=0, column=7, padx=6, pady=6)

        self.selection_label = ttk.Label(root, text="No images selected.", padding=(12, 0))
        self.selection_label.pack(anchor="w")

        columns = ("label", "confidence", "fire_probability", "model", "path")
        self.tree = ttk.Treeview(root, columns=columns, show="headings", height=20)
        self.tree.heading("label", text="Prediction")
        self.tree.heading("confidence", text="Confidence")
        self.tree.heading("fire_probability", text="Fire Prob")
        self.tree.heading("model", text="Model")
        self.tree.heading("path", text="Path")
        self.tree.column("label", width=110, anchor="center")
        self.tree.column("confidence", width=100, anchor="e")
        self.tree.column("fire_probability", width=100, anchor="e")
        self.tree.column("model", width=140, anchor="center")
        self.tree.column("path", width=500, anchor="w")
        self.tree.pack(fill="both", expand=True, padx=12, pady=12)

    def choose_images(self) -> None:
        files = filedialog.askopenfilenames(
            title="Choose images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff")],
        )
        if not files:
            return
        self.selected_paths = [Path(path) for path in files]
        self.selection_label.config(text=f"Selected {len(self.selected_paths)} image(s)")

    def choose_folder(self) -> None:
        folder = filedialog.askdirectory(title="Choose folder with images")
        if not folder:
            return
        self.selected_paths = [Path(folder)]
        self.selection_label.config(text=f"Selected folder: {folder}")

    def classify(self) -> None:
        if not self.selected_paths:
            messagebox.showwarning("WildFireDetector", "Choose at least one image or a folder first.")
            return

        try:
            threshold = float(self.threshold_var.get())
        except ValueError:
            messagebox.showerror("WildFireDetector", "Threshold must be a valid number.")
            return

        try:
            image_paths = collect_image_paths(self.selected_paths)
            if self.model_var.get() == "ensemble":
                rows = ensemble_predict(
                    image_paths,
                    threshold=threshold,
                    tta=self.tta_var.get(),
                )
            else:
                classifier = load_classifier(self.model_var.get())
                rows = predict_with_classifier(
                    classifier,
                    image_paths,
                    threshold=threshold,
                    tta=self.tta_var.get(),
                )
        except Exception as exc:
            messagebox.showerror("WildFireDetector", str(exc))
            return

        for item in self.tree.get_children():
            self.tree.delete(item)

        for row in rows:
            self.tree.insert(
                "",
                "end",
                values=(
                    row["predicted_label"],
                    f"{row['confidence']:.3f}",
                    f"{row['fire_probability']:.3f}",
                    row["model"],
                    row["path"],
                ),
            )

        self.selection_label.config(text=f"Classified {len(rows)} image(s)")


def main() -> None:
    if tk is None:
        raise RuntimeError(
            "Tkinter is not available in this Python environment. "
            "Use the CLI classifier or run the app with a Python build that includes Tk support."
        )

    root = tk.Tk()
    app = WildfireClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
