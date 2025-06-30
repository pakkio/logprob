# Sentence Confidence Analyzer

## Panoramica

Analizzare la confidenza di un modello linguistico nelle proprie risposte è fondamentale per identificare potenziali **allucinazioni** (informazioni non accurate) nei testi generati dall'intelligenza artificiale. Questo strumento nasce proprio per **visualizzare graficamente** il livello di sicurezza con cui modelli come GPT-4 generano ogni elemento del testo, permettendo così di individuare rapidamente passaggi potenzialmente problematici.

![image](https://github.com/user-attachments/assets/e4613415-9f14-4e5c-b66e-88ad8e0994b8)

## Caratteristiche principali

- **Analisi multi-livello**: Esamina la confidenza a livello di *token*, *parola* o *frase* intera
- **Visualizzazione cromatica intuitiva**: Codice colore da verde (alta confidenza) a rosso (bassa confidenza)
- **Compatibilità estesa**: Supporta GPT-4.1, GPT-4o, GPT-4o-mini, GPT-4-turbo e GPT-3.5-turbo
- **Implementazioni multiple**: Disponibile sia come app web Python (Gradio) che come pagina HTML standalone
- **Test di connessione API**: Verifica la validità della chiave API prima dell'analisi
- **Segmentazione intelligente del testo**: Riconoscimento avanzato di frasi, titoli ed elenchi
- **Legenda dettagliata**: Interpretazione chiara dei livelli di confidenza

## Come funziona

Il tool sfrutta le "probabilità logaritmiche" (logprobs) fornite dall'API OpenAI per calcolare quanto il modello sia "sicuro" delle parole generate. **Il processo di analisi** si articola così:

1. Invio di una richiesta all'API OpenAI con il parametro `logprobs=true`
2. Trasformazione dei valori di logprob in percentuali di confidenza mediante la formula `exp(logprob) * 100`
3. Segmentazione del testo secondo la granularità selezionata (token/parola/frase)
4. Calcolo della confidenza media per ogni segmento
5. Visualizzazione dei risultati con code colore appropriato

## Prerequisiti

- Un browser web moderno
- Una chiave API OpenAI valida
- Accesso a internet
- Per la versione Python: Python 3.6+ e pacchetti gradio, requests

## Installazione

### Versione HTML standalone
Nessuna installazione richiesta:
1. Salvare il codice HTML in un file con estensione `.html`
2. Aprire il file con un browser web

### Versione Python (Gradio)

Using Poetry (recommended):
```bash
poetry install --no-root
poetry run python logprob_gradio.py
```

Or using pip:
```bash
pip install gradio requests
python logprob_gradio.py
```

## Utilizzo

1. **Inserisci la tua chiave API**: Immetti la tua chiave OpenAI nell'apposito campo
2. **Verifica la connessione**: Usa il pulsante "Test Connessione" per verificare la validità della chiave
3. **Seleziona un modello**: Scegli il modello OpenAI da utilizzare (consigliato GPT-4o)
4. **Seleziona la granularità** (solo versione avanzata): Scegli se analizzare token, parole o frasi
5. **Inserisci il prompt**: Scrivi o incolla il testo del prompt nella casella
6. **Analizza**: Fai clic su "Analizza Confidenza" per avviare il processo

I risultati mostreranno ogni segmento della risposta con:
- Un **colore** indicante il livello di confidenza
- Una **percentuale** numerica di confidenza
- Un'**etichetta qualitativa** (Altissima, Alta, Media, Bassa, Molto bassa)

## Interpretazione dei risultati

- **Verde brillante (>95%)**: Il modello è estremamente sicuro di questo contenuto
- **Verde chiaro (85-95%)**: Alta confidenza, generalmente affidabile
- **Giallo/Arancione (70-85%)**: Media confidenza, potrebbero esserci imprecisioni
- **Arancione scuro (50-70%)**: Bassa confidenza, rischio elevato di imprecisioni
- **Rosso (<50%)**: Confidenza molto bassa, alta probabilità di allucinazioni

## Dettagli tecnici

### Calcolo della confidenza
La conversione da logprob a percentuale di confidenza avviene tramite:
```python
def logprob_to_confidence(logprob: float) -> float:
    """Converte logprob in percentuale di confidenza."""
    return math.exp(logprob) * 100
```

### Segmentazione intelligente del testo
La versione avanzata implementa un algoritmo sofisticato per identificare correttamente:
- **Titoli markdown** (linee che iniziano con ###)
- **Elementi di elenchi puntati** (linee che iniziano con -)
- **Frasi normali** (terminate da punto, esclamativo o interrogativo)

### Mappatura token-parole-frasi
Il sistema crea una mappatura precisa tra:
1. I token restituiti dall'API
2. Le parole nel testo
3. Le frasi o i segmenti identificati

## Differenze tra versioni

### Versione base (HTML standalone)
- Leggera e facile da usare
- Supporta solo l'analisi a livello di frase
- Non richiede installazione di pacchetti

### Versione avanzata (Python/Gradio)
- **Interfaccia più ricca** con più opzioni
- Supporta analisi a livello di token, parola o frase
- **Legenda cromatica** migliorata
- Riconoscimento avanzato dei segmenti di testo (titoli, elenchi)

## Casi d'uso

- **Verifica di contenuti scientifici**: Identifica quali parti di una risposta su argomenti scientifici potrebbero essere meno affidabili
- **Analisi di articoli storici**: Determina il livello di confidenza su specifiche affermazioni storiche
- **Confronto tra modelli**: Compara il livello di "certezza" di diversi modelli sulla stessa domanda
- **Ricerca e sviluppo AI**: Migliora la comprensione della calibrazione dei modelli linguistici
- **Didattica sull'IA**: Dimostra visivamente come i modelli valutano la propria confidenza

## Limitazioni

- Funziona solo con l'API OpenAI (non supporta altri provider come Anthropic Claude o Google Gemini)
- Richiede che il modello supporti il parametro `logprobs`
- Le percentuali di confidenza riflettono la certezza del modello, non necessariamente la precisione fattuale
- L'analisi a livello di frase è un'approssimazione e potrebbe non catturare sfumature a livello di singola parola
- Non tutti i modelli OpenAI supportano i logprobs allo stesso modo

## Sviluppi futuri

- Implementazione di un'esportazione dei risultati in formato CSV/JSON
- Analisi comparativa tra diversi modelli sullo stesso prompt
- Supporto per altri provider di API (quando renderanno disponibili le probabilità)
- Versione desktop standalone con Electron
- Integrazione con strumenti di scrittura e editing

## Autori

[Il tuo nome]

## Licenza

Questo tool è disponibile sotto licenza MIT.

## Riferimenti:

[1] OpenAI API Documentation: https://platform.openai.com/docs/api-reference/completions/create
[2] Gradio Documentation: https://www.gradio.app/docs/
[3] Anthropic Claude Technical Report: https://arxiv.org/abs/2304.01746
[4] Jiang, H.; Zhang, X.; Chen, M. (2023) "Analyzing Reliability in Modern Large Language Models" arXiv: 2306.09896
[5] Lin, S.; Jacob, A.P.; Porras, P. (2023) "Measuring Calibration in Large Language Models" arXiv: 2312.07466
