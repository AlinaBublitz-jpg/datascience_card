import tkinter as tk
from tkinter import ttk
import random

def get_psp_recommendation():
    """
    Diese Funktion nimmt Benutzereingaben für Betrag, Land und Uhrzeit und simuliert
    eine Vorhersage der Erfolgswahrscheinlichkeiten für verschiedene PSPs.
    """
    amount = amount_entry.get()
    country = country_var.get()
    hour = hour_var.get()

    # Simulierte Erfolgswahrscheinlichkeiten für verschiedene PSPs
    # Hier werden Zufallswerte verwendet. In einem realen Modell würden hier Modellvorhersagen eingesetzt.
    psp_probabilities = {
        "Moneycard": round(random.uniform(0.6, 0.9), 2),
        "Goldcard": round(random.uniform(0.5, 0.85), 2),
        "UK_Card": round(random.uniform(0.4, 0.8), 2),
        "Simplecard": round(random.uniform(0.7, 0.95), 2),
    }

    # Bestes PSP basierend auf der höchsten Erfolgswahrscheinlichkeit bestimmen
    best_psp = max(psp_probabilities, key=psp_probabilities.get)

    # Aktualisierung der UI-Labels mit den berechneten Wahrscheinlichkeiten
    for psp, prob in psp_probabilities.items():
        psp_labels[psp].config(text=f"{psp}: {prob * 100:.1f}%")

    recommendation_label.config(text=f"Empfohlener PSP: {best_psp}")

# Erstellung des Hauptfensters der GUI
root = tk.Tk()
root.title("PSP Auswahl System")
root.geometry("400x400")

# Frame für die Transaktionsdetails
frame = ttk.LabelFrame(root, text="Transaktionsdetails")
frame.pack(pady=10, padx=10, fill="both")

# Eingabefeld für den Transaktionsbetrag
ttk.Label(frame, text="Betrag (€):").grid(row=0, column=0, padx=5, pady=5)
amount_entry = ttk.Entry(frame)
amount_entry.grid(row=0, column=1, padx=5, pady=5)

# Auswahl des Landes über ein Dropdown-Menü
ttk.Label(frame, text="Land:").grid(row=1, column=0, padx=5, pady=5)
country_var = tk.StringVar()
country_dropdown = ttk.Combobox(frame, textvariable=country_var, values=["Deutschland", "UK", "Frankreich", "Spanien"], state="readonly")
country_dropdown.grid(row=1, column=1, padx=5, pady=5)
country_dropdown.current(0)

# Auswahl der Uhrzeit über ein Dropdown-Menü (Stunden von 0 bis 23)
ttk.Label(frame, text="Uhrzeit:").grid(row=2, column=0, padx=5, pady=5)
hour_var = tk.StringVar()
hour_dropdown = ttk.Combobox(frame, textvariable=hour_var, values=[str(i) for i in range(24)], state="readonly")
hour_dropdown.grid(row=2, column=1, padx=5, pady=5)
hour_dropdown.current(0)

# Frame zur Anzeige der PSP-Wahrscheinlichkeiten
psp_frame = ttk.LabelFrame(root, text="PSP Wahrscheinlichkeiten")
psp_frame.pack(pady=10, padx=10, fill="both")

psp_labels = {}
for idx, psp in enumerate(["Moneycard", "Goldcard", "UK_Card", "Simplecard"]):
    lbl = ttk.Label(psp_frame, text=f"{psp}: --")
    lbl.pack(anchor="w", padx=10, pady=2)
    psp_labels[psp] = lbl

# Label zur Anzeige der empfohlenen PSP-Auswahl
recommendation_label = ttk.Label(root, text="Empfohlener PSP: --", font=("Arial", 12, "bold"))
recommendation_label.pack(pady=10)

# Button zur Berechnung der PSP-Empfehlung
predict_button = ttk.Button(root, text="PSP Empfehlung berechnen", command=get_psp_recommendation)
predict_button.pack(pady=10)

root.mainloop()
