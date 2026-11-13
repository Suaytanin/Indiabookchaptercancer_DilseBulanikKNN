"""
DILSELRISKGORSELLESTIRME.py
Breast Cancer Early Diagnosis System — Linguistic Fuzzy Risk Approach
KNN (K=5, Threshold=0.40) + Triangular Membership Functions + Weighted Risk Index
+ Live Fuzzy Membership Visualization Panel
"""

import os
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use('TkAgg')  # [DÜZELTME] headless/backend sorunlarını önler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             recall_score, accuracy_score, f1_score, roc_auc_score)


# ============================================================
# FUZZY LOGIC — LINGUISTIC RISK SCORE
# ============================================================

def triangle_mf(x, a, b, c):
    """Triangular membership function (a=left base, b=peak, c=right base)."""
    # [DÜZELTME] x == b kontrolü en başa alındı (v1'deki doğru hâl)
    if x == b:
        return 1.0
    if x <= a or x >= c:
        return 0.0
    # [DÜZELTME] a == b veya b == c durumunda sıfıra bölme koruması
    if x < b:
        return (x - a) / (b - a) if b > a else 1.0
    return (c - x) / (c - b) if c > b else 1.0


def triangle_mf_vec(x, a, b, c):
    """Vectorized triangular membership function for plotting."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if b > a:
        idx = (x >= a) & (x < b)
        y[idx] = (x[idx] - a) / (b - a)
    y[x == b] = 1.0
    if c > b:
        idx = (x > b) & (x <= c)
        y[idx] = (c - x[idx]) / (c - b)
    return y


# Membership function parameters matching the manuscript (Section 4.3)
MF_LOW    = (0.00, 0.00, 0.40)
MF_MEDIUM = (0.15, 0.40, 0.65)
MF_HIGH   = (0.60, 0.85, 1.00)

RISK_VALUES = {"Low": 0.0, "Medium": 0.5, "High": 1.0}


def fuzzy_memberships(p):
    """Computes linguistic membership degrees for cancer probability p in [0, 1]."""
    p = float(np.clip(p, 0.0, 1.0))
    # [DÜZELTME] Low için açık sol-shoulder (peak 0.00 → P<=0'da μ=1)
    mu_low = 1.0 if p <= MF_LOW[1] else triangle_mf(p, *MF_LOW)
    # [DÜZELTME] High için sağ-shoulder saturasyon (P >= 0.85 → μ=1)
    mu_high = 1.0 if p >= MF_HIGH[1] else triangle_mf(p, *MF_HIGH)
    return {
        "Low":    mu_low,
        "Medium": triangle_mf(p, *MF_MEDIUM),
        "High":   mu_high,
    }


def fuzzy_risk_score(p):
    """Computes weighted linguistic risk index R in [0, 1]."""
    mu = fuzzy_memberships(p)
    denom = mu["Low"] + mu["Medium"] + mu["High"]
    if denom <= 1e-12:
        return float(p)
    return (RISK_VALUES["Low"] * mu["Low"]
            + RISK_VALUES["Medium"] * mu["Medium"]
            + RISK_VALUES["High"] * mu["High"]) / denom


def fuzzy_decision(p):
    """Maps weighted risk index R to three clinical decision categories."""
    R = fuzzy_risk_score(p)
    if R >= 0.50:
        return ("Malignant", R, "High risk: clinical intervention and further examination recommended.")
    if R >= 0.30:
        return ("Suspicious", R, "Moderate risk: repeat biopsy or close clinical follow-up recommended.")
    return ("Benign", R, "Low risk: routine follow-up recommended.")


# ============================================================
# GUI APPLICATION
# ============================================================

class KanserTeshisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Breast Cancer Early Detection System — Linguistic Fuzzy KNN")
        self.root.geometry("1400x820")
        self.root.configure(bg="#f4f6f9")

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TLabel", background="#f4f6f9", font=("Helvetica", 10))
        self.style.configure("TButton", font=("Helvetica", 10, "bold"), padding=6)

        self.model = None
        self.scaler = None
        self.optimal_k = 5
        self.custom_threshold = 0.40
        self.feature_names = []
        self.X_test_sample = None
        self.y_test = None

        self.fuzzy_canvas = None
        self.fuzzy_fig = None

        self.create_widgets()
        self.train_pipeline()
        self._draw_fuzzy_mf_plot(probability=None)

    def create_widgets(self):
        top_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        top_frame.pack(fill=tk.X)
        title_lbl = tk.Label(
            top_frame,
            text="AI-Based Clinical Decision Support System — Linguistic Fuzzy KNN",
            fg="white", bg="#2c3e50", font=("Helvetica", 15, "bold")
        )
        title_lbl.pack(pady=15)

        main_frame = tk.Frame(self.root, bg="#f4f6f9")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # -------- LEFT PANEL --------
        self.left_panel = tk.LabelFrame(
            main_frame,
            text=" Patient Cell Parameters (Biopsy Results) ",
            font=("Helvetica", 11, "bold"),
            bg="#ffffff", fg="#2c3e50", bd=2
        )
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)

        canvas = tk.Canvas(self.left_panel, bg="#ffffff", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.left_panel, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas, bg="#ffffff")
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # -------- RIGHT PANEL --------
        right_panel = tk.Frame(main_frame, bg="#f4f6f9")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=5)

        report_frame = tk.LabelFrame(
            right_panel, text=" Model Training and Performance Report ",
            font=("Helvetica", 11, "bold"),
            bg="#ffffff", fg="#2c3e50"
        )
        report_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.txt_report = tk.Text(
            report_frame, font=("Consolas", 9),
            bg="#ffffff", fg="#1e1e1e", bd=0, height=10
        )
        self.txt_report.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fuzzy_frame = tk.LabelFrame(
            right_panel,
            text=" Fuzzy Visualization — Linguistic Risk Membership Functions ",
            font=("Helvetica", 11, "bold"),
            bg="#ffffff", fg="#2c3e50"
        )
        self.fuzzy_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.graph_frame = tk.LabelFrame(
            right_panel, text=" Hyperparameter Evaluation Plots (K = 1..20) ",
            font=("Helvetica", 11, "bold"),
            bg="#ffffff", fg="#2c3e50"
        )
        self.graph_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # -------- BOTTOM PANEL --------
        bottom_frame = tk.Frame(self.root, bg="#ffffff", height=100, bd=1, relief=tk.SUNKEN)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM, ipady=10)

        btn_frame = tk.Frame(bottom_frame, bg="#ffffff")
        btn_frame.pack(side=tk.LEFT, padx=30, pady=10)

        btn_sample = ttk.Button(
            btn_frame, text="Load Random Test Patient",
            command=self.load_sample_patient
        )
        btn_sample.pack(side=tk.LEFT, padx=10)

        btn_predict = ttk.Button(
            btn_frame, text="Evaluate Linguistic Risk",
            command=self.predict_patient
        )
        btn_predict.pack(side=tk.LEFT, padx=10)

        self.lbl_result = tk.Label(
            bottom_frame, text="DIAGNOSIS: PENDING...",
            font=("Helvetica", 12, "bold"),
            bg="#e1e8ed", fg="#7f8c8d",
            width=62, height=3, relief=tk.RIDGE, wraplength=700
        )
        self.lbl_result.pack(side=tk.RIGHT, padx=30, pady=10)

    def _draw_fuzzy_mf_plot(self, probability=None):
        """Draws triangular membership functions and highlights current probability P."""
        if self.fuzzy_canvas is not None:
            self.fuzzy_canvas.get_tk_widget().destroy()
        if self.fuzzy_fig is not None:
            plt.close(self.fuzzy_fig)

        fig, ax = plt.subplots(figsize=(7, 2.4))
        fig.patch.set_facecolor('#ffffff')

        x = np.linspace(0, 1, 501)
        mu_low = triangle_mf_vec(x, *MF_LOW)
        mu_med = triangle_mf_vec(x, *MF_MEDIUM)
        # [DÜZELTME] High için sağ-shoulder saturasyon grafikte de uygulanıyor
        mu_high = np.where(x >= MF_HIGH[1], 1.0, triangle_mf_vec(x, *MF_HIGH))

        ax.plot(x, mu_low,  color='#2ecc71', linewidth=1.8, label='Low Risk')
        ax.plot(x, mu_med,  color='#f39c12', linewidth=1.8, label='Medium Risk')
        ax.plot(x, mu_high, color='#e74c3c', linewidth=1.8, label='High Risk')

        ax.fill_between(x, mu_low,  alpha=0.10, color='#2ecc71')
        ax.fill_between(x, mu_med,  alpha=0.10, color='#f39c12')
        ax.fill_between(x, mu_high, alpha=0.10, color='#e74c3c')

        if probability is not None:
            p = float(np.clip(probability, 0.0, 1.0))
            mu = fuzzy_memberships(p)
            R = fuzzy_risk_score(p)

            ax.axvline(x=p, color='#2c3e50', linestyle='--', linewidth=1.5,
                       label=f'P = {p:.3f}')

            ax.scatter([p], [mu['Low']],    color='#2ecc71', s=40, zorder=5)
            ax.scatter([p], [mu['Medium']], color='#f39c12', s=40, zorder=5)
            ax.scatter([p], [mu['High']],   color='#e74c3c', s=40, zorder=5)

            ax.text(
                0.02, 0.95,
                f"P={p:.3f}   μLow={mu['Low']:.2f}  μMed={mu['Medium']:.2f}  μHigh={mu['High']:.2f}   →  R={R:.3f}",
                transform=ax.transAxes, fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fdf6e3", edgecolor="#ccc"),
                verticalalignment='top'
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.08)
        ax.set_xlabel('Cancer Probability P', fontsize=8)
        ax.set_ylabel('Membership Degree μ', fontsize=8)
        ax.set_title('Linguistic Risk Membership Functions', fontsize=9, fontweight='bold')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        self.fuzzy_fig = fig
        self.fuzzy_canvas = FigureCanvasTkAgg(fig, master=self.fuzzy_frame)
        self.fuzzy_canvas.draw()
        self.fuzzy_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def train_pipeline(self):
        if not os.path.exists('data.csv'):
            messagebox.showerror(
                "Error",
                "The file 'data.csv' could not be found in the current directory!"
            )
            self.root.destroy()
            return

        df = pd.read_csv('data.csv')
        df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
        df.columns = df.columns.str.replace(' ', '_')
        self.feature_names = df.drop(columns=['diagnosis']).columns.tolist()

        self.entries = {}
        for idx, col in enumerate(self.feature_names):
            lbl = ttk.Label(self.scrollable_frame,
                            text=f"{col.replace('_', ' ').title()}:")
            lbl.grid(row=idx, column=0, sticky=tk.W, padx=10, pady=3)
            entry = ttk.Entry(self.scrollable_frame, width=15)
            entry.grid(row=idx, column=1, padx=10, pady=3)
            self.entries[col] = entry

        X = df.drop(columns=['diagnosis'])
        y = df['diagnosis'].map({'M': 1, 'B': 0})

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        self.X_test_sample = X_test
        self.y_test = y_test

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Evaluate K from 1 to 20 via 5-fold CV for plotting
        k_list = list(range(1, 21))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        cv_recall_scores = []
        cv_error_rates = []
        for k in k_list:
            knn_temp = KNeighborsClassifier(n_neighbors=k)
            rec = cross_val_score(knn_temp, X_train_scaled, y_train,
                                  cv=cv, scoring='recall').mean()
            acc = cross_val_score(knn_temp, X_train_scaled, y_train,
                                  cv=cv, scoring='accuracy').mean()
            cv_recall_scores.append(rec)
            cv_error_rates.append(1 - acc)

        # [NOT] K=5 ve thr=0.40 makaledeki gerekçeyle (0.20 posterior granularity) a priori seçildi.
        #       CV eğrileri bu seçimi görsel olarak doğrular; kod bunları sadece çizer.
        self.optimal_k = 5
        self.custom_threshold = 0.40

        self.model = KNeighborsClassifier(n_neighbors=self.optimal_k)
        self.model.fit(X_train_scaled, y_train)

        y_probs = self.model.predict_proba(X_test_scaled)[:, 1]
        y_pred_adj = (y_probs >= self.custom_threshold).astype(int)

        conf_matrix = confusion_matrix(y_test, y_pred_adj)
        accuracy = accuracy_score(y_test, y_pred_adj)
        recall = recall_score(y_test, y_pred_adj, zero_division=0)
        f1 = f1_score(y_test, y_pred_adj, zero_division=0)
        auc_val = roc_auc_score(y_test, y_probs)

        rep = classification_report(
            y_test, y_pred_adj,
            target_names=['Benign (B)', 'Malignant (M)'],
            zero_division=0
        )

        fuzzy_labels = [fuzzy_decision(p)[0] for p in y_probs]
        n_benign = fuzzy_labels.count("Benign")
        n_susp = fuzzy_labels.count("Suspicious")
        n_mal = fuzzy_labels.count("Malignant")

        # [DÜZELTME] report_text stringleri v1'deki gibi \n escape ile temiz hâle getirildi
        report_text  = f"[INFO] Selected Neighbor Count (K)     : {self.optimal_k}\n"
        report_text += f"[INFO] Decision Threshold (classic)    : {self.custom_threshold:.2f}\n\n"
        report_text += f"── Classic Test Set Performance (n = {len(y_test)}) ──\n"
        report_text += f"Accuracy : %{accuracy*100:.2f}\n"
        report_text += f"Recall   : %{recall*100:.2f}\n"
        report_text += f"F1 Score : %{f1*100:.2f}\n"
        report_text += f"AUC-ROC  : {auc_val:.4f}\n\n"
        report_text += f"Confusion Matrix:\n{conf_matrix}\n\n"
        report_text += f"── Linguistic Fuzzy Risk — Test Set Distribution ──\n"
        report_text += f"Benign     (R < 0.30)        : {n_benign}\n"
        report_text += f"Suspicious (0.30 <= R < 0.50): {n_susp}\n"
        report_text += f"Malignant  (R >= 0.50)       : {n_mal}\n\n"
        report_text += f"Detailed Metric Report:\n{rep}"

        self.txt_report.config(state=tk.NORMAL)
        self.txt_report.delete("1.0", tk.END)
        self.txt_report.insert(tk.END, report_text)
        self.txt_report.config(state=tk.DISABLED)

        # Hyperparameter plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.1))
        fig.patch.set_facecolor('#ffffff')

        ax1.plot(k_list, cv_error_rates, color='red', linestyle='--',
                 marker='o', markersize=4)
        ax1.set_title('CV Error Rate (1 - Acc)', fontsize=8)
        ax1.set_xlabel('K', fontsize=7)
        ax1.grid(True, alpha=0.3)

        ax2.plot(k_list, cv_recall_scores, color='green', linestyle='-',
                 marker='s', markersize=4)
        ax2.axvline(x=self.optimal_k, color='orange', linestyle='--',
                    linewidth=1, label=f'Selected K={self.optimal_k}')
        ax2.set_title('CV Recall', fontsize=8)
        ax2.set_xlabel('K', fontsize=7)
        ax2.legend(fontsize=6)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        canvas_graph = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas_graph.draw()
        canvas_graph.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        plt.close(fig)

        # [DÜZELTME] Doğrulama için konsola da yaz (makale Tablo 2 ile karşılaştırma amaçlı)
        print("=" * 60)
        print("CM:", conf_matrix.tolist())
        print(f"Accuracy: {accuracy:.4f} | Recall(malignant): {recall:.4f} | "
              f"F1(malignant): {f1:.4f} | AUC: {auc_val:.4f}")
        print(f"Fuzzy distribution — Benign: {n_benign} | Suspicious: {n_susp} | Malignant: {n_mal}")
        print("=" * 60)

    def load_sample_patient(self):
        if self.X_test_sample is None:
            return
        random_idx = np.random.choice(self.X_test_sample.index)
        sample_data = self.X_test_sample.loc[random_idx]
        for col in self.feature_names:
            self.entries[col].delete(0, tk.END)
            self.entries[col].insert(0, f"{sample_data[col]:.4f}")

    def predict_patient(self):
        if self.model is None or self.scaler is None:
            return

        try:
            raw_features = []
            for col in self.feature_names:
                txt = self.entries[col].get().strip()
                if not txt:
                    raise ValueError(f"Empty field: {col}")
                val = float(txt)
                if not np.isfinite(val):
                    raise ValueError(f"Invalid number: {col} = {txt}")
                raw_features.append(val)

            patient_df = pd.DataFrame([raw_features], columns=self.feature_names)
            patient_scaled = self.scaler.transform(patient_df)

            probability = float(self.model.predict_proba(patient_scaled)[0, 1])

            label, R, msg = fuzzy_decision(probability)
            mu = fuzzy_memberships(probability)
            mu_txt = (f"μ(Low)={mu['Low']:.2f}  "
                      f"μ(Med)={mu['Medium']:.2f}  "
                      f"μ(High)={mu['High']:.2f}")

            if label == "Malignant":
                self.lbl_result.config(
                    text=(f"MALIGNANT | Fuzzy Risk R={R:.2f} | P={probability:.2f}\n"
                          f"{mu_txt}\n{msg}"),
                    bg="#ffcccc", fg="#cc0000"
                )
            elif label == "Suspicious":
                self.lbl_result.config(
                    text=(f"SUSPICIOUS | Fuzzy Risk R={R:.2f} | P={probability:.2f}\n"
                          f"{mu_txt}\n{msg}"),
                    bg="#fff3cd", fg="#856404"
                )
            else:
                self.lbl_result.config(
                    text=(f"BENIGN | Fuzzy Risk R={R:.2f} | P={probability:.2f}\n"
                          f"{mu_txt}\n{msg}"),
                    bg="#d4edda", fg="#155724"
                )

            self._draw_fuzzy_mf_plot(probability=probability)

        except ValueError as e:
            messagebox.showwarning(
                "Input Error",
                f"Please ensure valid numerical values are entered.\n\nDetail: {e}"
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = KanserTeshisApp(root)
    root.mainloop()