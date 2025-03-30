import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random
from io import BytesIO

# Navigation (Seitenauswahl)
st.set_page_config(page_title="PSP Dashboard", layout="centered")
st.sidebar.title("🧭 Navigation")
seite = st.sidebar.selectbox("Seite wählen", ["📈 Wirkung des Modells", "💳 Empfehlungssystem"])

# ----------------------------------
# Seite: Wirkung des Modells
# ----------------------------------
if seite == "📈 Wirkung des Modells":
    st.title("📈 Wirkung des Modells")

    # Dummy-Daten vorbereiten
    monate = pd.date_range(start="2023-01-01", periods=12, freq='ME').strftime('%b %Y')
    erfolgsrate_vorher = [round(random.uniform(0.65, 0.7), 2) for _ in range(12)]
    erfolgsrate_nachher = [round(e + random.uniform(0.05, 0.1), 2) for e in erfolgsrate_vorher]
    kosten_vorher = [round(random.uniform(4.4, 4.7), 2) for _ in range(12)]
    kosten_nachher = [round(k - random.uniform(0.4, 0.7), 2) for k in kosten_vorher]

    daten = pd.DataFrame({
        "Monat": monate,
        "Erfolgsrate vorher": erfolgsrate_vorher,
        "Erfolgsrate nachher": erfolgsrate_nachher,
        "Kosten vorher": kosten_vorher,
        "Kosten nachher": kosten_nachher
    })

    # Interaktiver Zeitraumfilter
    st.sidebar.header("📅 Zeitraum filtern")
    start_index = st.sidebar.selectbox("Von Monat", list(daten.index), format_func=lambda i: daten['Monat'][i])
    end_index = st.sidebar.selectbox("Bis Monat", list(daten.index), index=11, format_func=lambda i: daten['Monat'][i])

    gefilterte_daten = daten.iloc[start_index:end_index + 1]

    # Erfolgsrate anzeigen
    st.subheader("🟢 Erfolgsrate der Transaktionen")
    st.line_chart(gefilterte_daten.set_index("Monat")[['Erfolgsrate vorher', 'Erfolgsrate nachher']])

    # Transaktionskosten anzeigen
    st.subheader("💰 Durchschnittliche Transaktionskosten")
    st.line_chart(gefilterte_daten.set_index("Monat")[['Kosten vorher', 'Kosten nachher']])

    # Zeitersparnis anzeigen
    st.subheader("⏱️ Zeitersparnis durch Automatisierung")
    anzahl_transaktionen = 1000
    zeit_pro_transaktion_alt = 30  # Sekunden
    zeit_gespart_min = (anzahl_transaktionen * zeit_pro_transaktion_alt) / 60
    zeit_gespart_h = zeit_gespart_min / 60
    st.metric("Gesparte Zeit", f"{zeit_gespart_h:.1f} Stunden", delta="im Vergleich zur manuellen Auswahl")

    # Export-Funktion
    st.subheader("⬇️ Exportiere ausgewertete Daten")
    excel_buffer = BytesIO()
    gefilterte_daten.to_excel(excel_buffer, index=False)
    excel_buffer.seek(0)
    st.download_button(
        label="📥 Daten als Excel herunterladen",
        data=excel_buffer,
        file_name="psp_modellwirkung.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Modellinfo-Box
    with st.expander("ℹ️ Modellinfo: Auswahllogik & KI-Einsatz"):
        st.markdown("""
        **Modellbeschreibung:**
        Das System verwendet ein Support Vector Machine (SVM) Modell zur Vorhersage der Erfolgswahrscheinlichkeit je PSP.
        Die Auswahl folgt einer regelbasierten Logik:
        
        1. Wenn alle Wahrscheinlichkeiten 0 sind → *Simplecard*
        2. Wenn *Simplecard* die beste oder fast beste ist → *Simplecard*
        3. Dann *UK_Card*, *Moneycard*, *Goldcard* entsprechend der Erfolgswahrscheinlichkeit
        
        **Nutzen:**
        - Höhere Erfolgsquote
        - Niedrigere Transaktionskosten
        - Automatisierte Entscheidung → spart Personalzeit und reduziert Fehler
        """)

# ----------------------------------
# Seite: Empfehlungssystem
# ----------------------------------
elif seite == "💳 Empfehlungssystem":
    st.title("💳 Empfehlungssystem")

    amount = st.number_input("Betrag (€)", min_value=0.0, max_value=10000.0, step=1.0, value=100.0)
    country = st.selectbox("Land", ["Deutschland", "UK", "Frankreich", "Spanien"])
    hour = st.selectbox("Uhrzeit (Stunde)", [str(i) for i in range(24)], index=12)

    if st.button("📊 PSP Empfehlung berechnen"):
        psp_success_probabilities = {
            "Moneycard": round(random.uniform(0.6, 0.9), 2),
            "Goldcard": round(random.uniform(0.5, 0.85), 2),
            "UK_Card": round(random.uniform(0.4, 0.8), 2),
            "Simplecard": round(random.uniform(0.7, 0.95), 2),
        }

        all_probs = list(psp_success_probabilities.values())
        chosen_psp = None

        if all(prob == 0 for prob in all_probs):
            chosen_psp = 'Simplecard'
        else:
            max_prob = max(all_probs)
            if 'Simplecard' in psp_success_probabilities:
                simplecard_prob = psp_success_probabilities['Simplecard']
                if simplecard_prob == max_prob or max_prob - simplecard_prob < 0.1:
                    chosen_psp = 'Simplecard'
            if chosen_psp is None and 'UK_Card' in psp_success_probabilities:
                uk_card_prob = psp_success_probabilities['UK_Card']
                if uk_card_prob == max_prob or max_prob - uk_card_prob < 0.1:
                    chosen_psp = 'UK_Card'
            if chosen_psp is None and 'Moneycard' in psp_success_probabilities:
                moneycard_prob = psp_success_probabilities['Moneycard']
                if moneycard_prob == max_prob or max_prob - moneycard_prob < 0.1:
                    chosen_psp = 'Moneycard'
            if chosen_psp is None and 'Goldcard' in psp_success_probabilities:
                goldcard_prob = psp_success_probabilities['Goldcard']
                if goldcard_prob == max_prob:
                    chosen_psp = 'Goldcard'

        st.subheader("📊 Erfolgswahrscheinlichkeiten je PSP")
        df = pd.DataFrame.from_dict(psp_success_probabilities, orient='index', columns=['Success Probability'])
        df['Success Probability (%)'] = df['Success Probability'] * 100
        st.bar_chart(df['Success Probability (%)'])

        st.success(f"✅ Empfohlener PSP: **{chosen_psp}**")

        if st.button("🚀 Transaktion freigeben"):
            st.success(f"🎉 Transaktion wurde mit **{chosen_psp}** freigegeben!")