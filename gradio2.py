!pip install gradio
import gradio as gr
import requests
import re
import math
from typing import List, Dict, Any, Union

# Funzioni di utilità
def logprob_to_confidence(logprob: float) -> float:
    """Converte logprob in percentuale di confidenza."""
    return math.exp(logprob) * 100

def get_confidence_label(confidence: float) -> str:
    """Restituisce l'etichetta del livello di confidenza."""
    if confidence > 95:
        return "Altissima confidenza"
    if confidence > 85:
        return "Alta confidenza"
    if confidence > 70:
        return "Media confidenza"
    if confidence > 50:
        return "Bassa confidenza"
    return "Molto bassa confidenza"

def get_confidence_color(confidence: float) -> str:
    """Restituisce il colore basato sulla confidenza (da rosso a verde)."""
    r = max(0, min(255, round(255 * (1 - confidence / 100))))
    g = max(0, min(255, round(255 * (confidence / 100))))
    b = 0
    return f"rgb({r}, {g}, {b})"

# APPROCCIO COMPLETAMENTE NUOVO PER LA SEGMENTAZIONE DEL TESTO
def segment_text(text: str) -> List[str]:
    """Versione completamente rivista per segmentare il testo in frasi."""
    
    # Rimuovi spazi extra e normalizza il testo
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Definisci pattern di terminazione frase
    end_patterns = [
        r'(?<=[.!?])\s+(?=[A-Z])',  # Punto/esclamativo/interrogativo seguito da spazio e maiuscola
        r'(?<=\.)\s+(?=\-)',  # Punto seguito da trattino (per elenchi)
        r'(?<=\.)\s+(?=\*\*)',  # Punto seguito da asterischi (per formattazione markdown)
        r'(?<=\.)\s+(?=###)',  # Punto seguito da ### (per titoli markdown)
        r'(?<=\n)(?=###)',  # Newline seguito da ### (per titoli markdown)
        r'(?<=\n)(?=\-)',  # Newline seguito da trattino (per elenchi)
    ]
    
    # Combina i pattern
    combined_pattern = '|'.join(end_patterns)
    
    # Dividi il testo
    segments = re.split(combined_pattern, text)
    
    # Gestione speciale per titoli e sottotitoli
    refined_segments = []
    for segment in segments:
        # Cerca titoli nel segmento
        title_matches = re.findall(r'###\s+[^\n]+', segment)
        
        if title_matches:
            # Se ci sono titoli, trattali come segmenti separati
            remaining = segment
            for title in title_matches:
                parts = remaining.split(title, 1)
                if parts[0].strip():
                    refined_segments.append(parts[0].strip())
                refined_segments.append(title.strip())
                remaining = parts[1] if len(parts) > 1 else ""
            if remaining.strip():
                refined_segments.append(remaining.strip())
        else:
            # Altrimenti, aggiungi l'intero segmento
            if segment.strip():
                refined_segments.append(segment.strip())
    
    # Gestione speciale per elenchi puntati
    final_segments = []
    for segment in refined_segments:
        # Trova elenchi puntati
        list_items = re.findall(r'\-\s+[^\n\-]+', segment)
        
        if list_items and not segment.startswith('-'):
            # Se ci sono elementi di elenco ma non è un elenco completo, dividi
            parts = []
            current = segment
            for item in list_items:
                item_parts = current.split(item, 1)
                if item_parts[0].strip():
                    parts.append(item_parts[0].strip())
                parts.append(item.strip())
                current = item_parts[1] if len(item_parts) > 1 else ""
            if current.strip():
                parts.append(current.strip())
            final_segments.extend(parts)
        else:
            final_segments.append(segment)
    
    return [s for s in final_segments if s.strip()]

def extract_words(text: str) -> List[str]:
    """Estrae le parole dal testo in modo più robusto."""
    # Rimuove la punteggiatura eccetto trattini interni alle parole
    cleaned_text = re.sub(r'[^\w\s\-]|(?<!\w)\-|\-(?!\w)', ' ', text)
    words = [w for w in cleaned_text.split() if w.strip()]
    return words

# NUOVO SISTEMA DI MAPPATURA TOKEN-PAROLE-FRASI
def create_confidence_analysis(text: str, tokens: List[Dict[str, Any]], granularity: str) -> Dict[str, Any]:
    """Crea un'analisi della confidenza con un approccio completamente nuovo."""
    
    # Ottieni il testo completo dai token
    full_text = "".join(token['token'] for token in tokens)
    
    # Aggiungi la confidenza a ciascun token
    for token in tokens:
        token['confidence'] = logprob_to_confidence(token['logprob'])
    
    if granularity == "token":
        # Ottieni i segmenti 
        segments = segment_text(full_text)
        segment_map = []
        
        # Per ogni segmento, mappa i token
        token_offset = 0
        for segment in segments:
            segment_tokens = []
            segment_len = len(segment)
            current_len = 0
            
            while token_offset < len(tokens) and current_len < segment_len:
                token = tokens[token_offset]
                token_text = token['token']
                segment_tokens.append(token)
                current_len += len(token_text)
                token_offset += 1
                
                # Verifica se abbiamo superato la lunghezza del segmento
                if current_len >= segment_len:
                    break
            
            if segment_tokens:
                segment_map.append({
                    'text': segment,
                    'tokens': segment_tokens,
                })
        
        return {
            'text': full_text,
            'segments': segment_map,
            'granularity': 'token'
        }
        
    elif granularity == "word":
        # Ottieni i segmenti
        segments = segment_text(full_text)
        words = extract_words(full_text)
        
        # Crea indici di parole
        word_indices = []
        word_start_index = 0
        
        for word in words:
            # Trova la prossima occorrenza della parola
            word_pos = full_text[word_start_index:].lower().find(word.lower())
            if word_pos != -1:
                word_pos += word_start_index
                word_end = word_pos + len(word)
                word_indices.append((word, word_pos, word_end))
                word_start_index = word_end
        
        # Mappa i token alle parole
        word_data = []
        for word, start_pos, end_pos in word_indices:
            word_tokens = []
            token_offset = 0
            token_pos = 0
            
            for i, token in enumerate(tokens):
                token_start = token_pos
                token_end = token_start + len(token['token'])
                
                # Verifica se il token si sovrappone alla parola
                if (token_start <= end_pos and token_end >= start_pos):
                    word_tokens.append(token)
                
                token_pos += len(token['token'])
                
                # Se abbiamo superato la posizione di fine della parola, fermati
                if token_start > end_pos:
                    break
            
            if word_tokens:
                avg_logprob = sum(token['logprob'] for token in word_tokens) / len(word_tokens)
                confidence = logprob_to_confidence(avg_logprob)
                
                word_data.append({
                    'text': word,
                    'confidence': confidence,
                    'start': start_pos,
                    'end': end_pos
                })
        
        # Mappa le parole ai segmenti
        segment_map = []
        for segment in segments:
            segment_start = full_text.find(segment)
            segment_end = segment_start + len(segment)
            
            segment_words = []
            for word_info in word_data:
                # Verifica se la parola è all'interno del segmento
                if word_info['start'] >= segment_start and word_info['end'] <= segment_end:
                    segment_words.append(word_info)
            
            if segment_words:
                segment_map.append({
                    'text': segment,
                    'words': segment_words
                })
        
        return {
            'text': full_text,
            'segments': segment_map,
            'granularity': 'word'
        }
        
    else:  # sentence
        # Ottieni i segmenti
        segments = segment_text(full_text)
        segment_data = []
        
        # Per ogni segmento, calcola la confidenza media
        token_offset = 0
        for segment in segments:
            segment_start = full_text.find(segment)
            segment_end = segment_start + len(segment)
            
            segment_tokens = []
            token_pos = 0
            
            for token in tokens:
                token_start = token_pos
                token_end = token_start + len(token['token'])
                
                # Verifica se il token si sovrappone al segmento
                if (token_start <= segment_end and token_end >= segment_start):
                    segment_tokens.append(token)
                
                token_pos += len(token['token'])
                
                # Se abbiamo superato la posizione di fine del segmento, fermati
                if token_start > segment_end:
                    break
            
            if segment_tokens:
                avg_logprob = sum(token['logprob'] for token in segment_tokens) / len(segment_tokens)
                confidence = logprob_to_confidence(avg_logprob)
                
                segment_data.append({
                    'text': segment,
                    'confidence': confidence
                })
        
        return {
            'text': full_text,
            'segments': segment_data,
            'granularity': 'sentence'
        }

# Funzioni API e analisi
def test_api_connection(api_key: str) -> str:
    """Testa la connessione all'API OpenAI."""
    if not api_key:
        return "❌ Inserisci una API key prima di testare"
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        response = requests.get("https://api.openai.com/v1/models", headers=headers)
        
        if response.status_code == 200:
            return "✅ Connessione riuscita! API key valida."
        else:
            error_data = response.json()
            error_message = error_data.get('error', {}).get('message', response.reason)
            return f"❌ Errore API: {error_message}"
    except Exception as e:
        return f"❌ Errore di connessione: {str(e)}"

def analyze_confidence(api_key: str, model: str, prompt: str, granularity: str) -> Union[Dict[str, Any], Dict[str, str]]:
    """Analizza la confidenza del testo generato dal modello."""
    if not api_key:
        return {"error": "Inserisci una API key valida"}
    
    if not prompt:
        return {"error": "Inserisci un prompt"}
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant providing accurate and detailed information."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "logprobs": True,
            "top_logprobs": 1
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            error_data = response.json()
            error_message = error_data.get('error', {}).get('message', response.reason)
            return {"error": f"Errore API: {error_message}"}
        
        data = response.json()
        
        text = data['choices'][0]['message']['content']
        
        # Verifica se logprobs sono disponibili
        if 'logprobs' not in data['choices'][0] or 'content' not in data['choices'][0]['logprobs']:
            return {"error": "Logprobs non disponibili. Potrebbe essere dovuto alle limitazioni dell'API."}
        
        logprobs = data['choices'][0]['logprobs']['content']
        
        # Usa la nuova funzione unificata per l'analisi
        return create_confidence_analysis(text, logprobs, granularity)
            
    except Exception as e:
        return {"error": f"Errore: {str(e)}"}

# VISUALIZZAZIONE MIGLIORATA
def format_results(result: Union[Dict[str, Any], Dict[str, str]]) -> str:
    """Formatta i risultati dell'analisi in HTML."""
    if isinstance(result, dict) and "error" in result:
        return f"<div style='color: red; padding: 10px; border-left: 4px solid red; background-color: #ffeeee;'><strong>Errore:</strong> {result['error']}</div>"
    
    html = "<h2>Risultati dell'analisi</h2>"
    
    if result['granularity'] == 'token':
        html += "<h3>Analisi a livello di token</h3>"
        
        for i, segment in enumerate(result.get('segments', [])):
            # Determina il tipo di segmento (titolo, elenco, testo normale)
            is_title = segment['text'].startswith('###')
            is_list_item = segment['text'].startswith('-')
            
            if is_title:
                html += f"<div style='margin: 20px 0 10px 0; padding: 10px; background-color: #e9ecef; border-radius: 5px; font-weight: bold;'>"
            elif is_list_item:
                html += f"<div style='margin: 5px 0 5px 20px; padding: 5px 10px; background-color: #f8f9fa; border-left: 3px solid #6c757d;'>"
            else:
                html += f"<div style='margin: 15px 0; padding: 10px; border-radius: 5px; background-color: #f8f9fa; border-left: 4px solid #6c757d;'>"
            
            if not is_title and not is_list_item:
                html += f"<strong>Segmento {i + 1}:</strong> "
            
            for token in segment.get('tokens', []):
                color = get_confidence_color(token['confidence'])
                confidence_pct = f"{token['confidence']:.1f}%"
                label = get_confidence_label(token['confidence'])
                
                html += f"<span title='{confidence_pct} - {label}' style='color: {color}; font-weight: 600;'>{token['token']}</span>"
            
            html += "</div>"
    
    elif result['granularity'] == 'word':
        html += "<h3>Analisi a livello di parola</h3>"
        
        for i, segment in enumerate(result.get('segments', [])):
            # Determina il tipo di segmento
            is_title = segment['text'].startswith('###')
            is_list_item = segment['text'].startswith('-')
            
            if is_title:
                html += f"<div style='margin: 20px 0 10px 0; padding: 10px; background-color: #e9ecef; border-radius: 5px; font-weight: bold;'>"
            elif is_list_item:
                html += f"<div style='margin: 5px 0 5px 20px; padding: 5px 10px; background-color: #f8f9fa; border-left: 3px solid #6c757d;'>"
            else:
                html += f"<div style='margin: 15px 0; padding: 10px; border-radius: 5px; background-color: #f8f9fa; border-left: 4px solid #6c757d;'>"
            
            if not is_title and not is_list_item:
                html += f"<strong>Segmento {i + 1}:</strong> "
            
            # Ottieni il testo del segmento
            segment_text = segment['text']
            word_infos = segment.get('words', [])
            
            # Se non ci sono informazioni sulle parole, mostra il testo normale
            if not word_infos:
                html += segment_text
            else:
                # Costruisci una mappa delle posizioni delle parole
                word_map = {}
                for word_info in word_infos:
                    # Trova la posizione relativa all'interno del segmento
                    word_text = word_info['text']
                    segment_start = result['text'].find(segment_text)
                    relative_start = word_info['start'] - segment_start
                    
                    if relative_start >= 0 and relative_start < len(segment_text):
                        color = get_confidence_color(word_info['confidence'])
                        confidence_pct = f"{word_info['confidence']:.1f}%"
                        label = get_confidence_label(word_info['confidence'])
                        
                        word_map[relative_start] = {
                            'text': word_text,
                            'color': color,
                            'title': f"{confidence_pct} - {label}"
                        }
                
                # Ricostruisci il testo con le parole colorate
                if word_map:
                    positions = sorted(word_map.keys())
                    last_pos = 0
                    colored_text = ""
                    
                    for pos in positions:
                        # Aggiungi il testo prima della parola
                        if pos > last_pos:
                            colored_text += segment_text[last_pos:pos]
                        
                        # Aggiungi la parola colorata
                        word_info = word_map[pos]
                        word_len = len(word_info['text'])
                        
                        colored_text += f"<span title='{word_info['title']}' style='color: {word_info['color']}; font-weight: 600;'>{segment_text[pos:pos+word_len]}</span>"
                        
                        last_pos = pos + word_len
                    
                    # Aggiungi il testo rimanente
                    if last_pos < len(segment_text):
                        colored_text += segment_text[last_pos:]
                    
                    html += colored_text
                else:
                    html += segment_text
            
            html += "</div>"
            
    else:  # sentence
        html += "<h3>Analisi a livello di frase</h3>"
        
        for i, segment in enumerate(result.get('segments', [])):
            # Determina il tipo di segmento
            is_title = segment['text'].startswith('###')
            is_list_item = segment['text'].startswith('-')
            
            color = get_confidence_color(segment['confidence'])
            label = get_confidence_label(segment['confidence'])
            
            if is_title:
                html += f"""
                <div style='margin: 20px 0 10px 0; padding: 10px; background-color: #e9ecef; border-radius: 5px; 
                     color: {color}; border-left: 4px solid {color}; font-weight: bold;'>
                    {segment['text']}
                    <span style='margin-left: 10px; font-size: 0.9em;'>({segment['confidence']:.2f}% - {label})</span>
                </div>"""
            elif is_list_item:
                html += f"""
                <div style='margin: 5px 0 5px 20px; padding: 5px 10px; background-color: #f8f9fa; 
                     color: {color}; border-left: 4px solid {color};'>
                    {segment['text']}
                    <span style='margin-left: 10px; font-size: 0.9em;'>({segment['confidence']:.2f}% - {label})</span>
                </div>"""
            else:
                html += f"""
                <div style='margin: 10px 0; padding: 10px; border-radius: 4px; color: {color}; 
                     background-color: {color}15; border-left: 4px solid {color};'>
                    <strong>Segmento {i + 1}:</strong> {segment['text']}
                    <span style='margin-left: 10px; font-size: 0.9em;'>({segment['confidence']:.2f}% - {label})</span>
                </div>"""
    
    # LEGENDA MIGLIORATA CON BLOCCHI COLORATI
    html += """
    <div style='margin-top: 20px; padding: 15px; background-color: #f0f0f0; border-radius: 8px; border: 1px solid #ddd;'>
        <h4 style='margin-top: 0; margin-bottom: 10px;'>Legenda Confidenza:</h4>
        <div style='display: flex; flex-direction: column; gap: 8px;'>
            <div style='display: flex; align-items: center;'>
                <span style='display: inline-block; width: 25px; height: 25px; background-color: rgb(0, 255, 0); margin-right: 10px; border: 1px solid #333;'></span>
                <span style='font-weight: bold;'>Altissima confidenza (95-100%)</span>
            </div>
            <div style='display: flex; align-items: center;'>
                <span style='display: inline-block; width: 25px; height: 25px; background-color: rgb(85, 170, 0); margin-right: 10px; border: 1px solid #333;'></span>
                <span style='font-weight: bold;'>Alta confidenza (85-95%)</span>
            </div>
            <div style='display: flex; align-items: center;'>
                <span style='display: inline-block; width: 25px; height: 25px; background-color: rgb(170, 85, 0); margin-right: 10px; border: 1px solid #333;'></span>
                <span style='font-weight: bold;'>Media confidenza (70-85%)</span>
            </div>
            <div style='display: flex; align-items: center;'>
                <span style='display: inline-block; width: 25px; height: 25px; background-color: rgb(210, 45, 0); margin-right: 10px; border: 1px solid #333;'></span>
                <span style='font-weight: bold;'>Bassa confidenza (50-70%)</span>
            </div>
            <div style='display: flex; align-items: center;'>
                <span style='display: inline-block; width: 25px; height: 25px; background-color: rgb(255, 0, 0); margin-right: 10px; border: 1px solid #333;'></span>
                <span style='font-weight: bold;'>Molto bassa confidenza (0-50%)</span>
            </div>
        </div>
    </div>
    """
    
    return html

# Funzioni per gli indicatori di caricamento
def start_processing():
    return "<div style='display: flex; align-items: center; margin-top: 10px;'><div style='width: 20px; height: 20px; border-radius: 50%; border: 3px solid #3498db; border-top-color: transparent; animation: spin 1s linear infinite; margin-right: 10px;'></div><span style='color: #3498db;'><strong>Elaborazione in corso...</strong> Sto analizzando il testo con il modello selezionato</span></div><style>@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }</style>"

def end_processing():
    return "<div style='color: #28a745; margin-top: 10px;'><strong>✓ Analisi completata!</strong></div>"

# Funzioni per Gradio
def run_analysis(api_key: str, model: str, prompt: str, granularity: str) -> str:
    """Funzione principale per l'analisi."""
    result = analyze_confidence(api_key, model, prompt, granularity)
    return format_results(result)

# Interfaccia Gradio
def create_interface():
    with gr.Blocks(title="Sentence Confidence Analyzer Pro") as app:
        gr.Markdown("# Sentence Confidence Analyzer Pro")
        
        with gr.Group():
            gr.Markdown("## OpenAI API Key")
            
            api_key_input = gr.Textbox(
                type="password",
                label="API Key",
                placeholder="Inserisci la tua API key di OpenAI (sk-...)"
            )
            
            test_btn = gr.Button("Test Connessione")
            connection_status = gr.Markdown("")
            
            # Collegamento per il test della connessione
            test_btn.click(
                test_api_connection,
                inputs=[api_key_input],
                outputs=[connection_status]
            )
        
        with gr.Group():
            gr.Markdown("## Impostazioni Analisi")
            
            with gr.Row():
                with gr.Column():
                    model_select = gr.Dropdown(
                        choices=[
                            "gpt-4.1",
                            "gpt-4o-2024-08-06",
                            "gpt-4o-mini-2024-07-18",
                            "gpt-4-turbo-2024-04-09",
                            "gpt-3.5-turbo-0125"
                        ],
                        value="gpt-4o-2024-08-06",
                        label="Modello"
                    )
                
                with gr.Column():
                    granularity_select = gr.Radio(
                        choices=["token", "word", "sentence"],
                        value="sentence",
                        label="Granularità di analisi",
                        info="Scegli a che livello analizzare la confidenza"
                    )
        
        with gr.Group():
            gr.Markdown("## Prompt")
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Inserisci il tuo prompt",
                value="Descrivi le tecniche di scavo e i principali ritrovamenti archeologici del sito di Göbekli Tepe in Turchia, confrontandoli con quelli di Gunung Padang in Indonesia.",
                lines=5
            )
        
        # Migliorato indicatore di elaborazione
        with gr.Group():
            analyze_btn = gr.Button("Analizza Confidenza", variant="primary", size="large")
            
            with gr.Row():
                status_indicator = gr.Markdown("")
                
        # Risultati
        results_html = gr.HTML()
        
        # Colleghiamo l'analisi con gli indicatori di caricamento migliorati
        analyze_btn.click(
            fn=start_processing,
            inputs=None,
            outputs=status_indicator
        ).then(
            fn=run_analysis,
            inputs=[api_key_input, model_select, prompt_input, granularity_select],
            outputs=results_html,
            show_progress=True
        ).then(
            fn=end_processing,
            inputs=None,
            outputs=status_indicator
        )
        
        # Aggiungiamo una sezione informativa
        with gr.Accordion("Informazioni sull'app", open=False):
            gr.Markdown("""
            ## Come funziona questa app
            
            Questa applicazione analizza il livello di confidenza del modello OpenAI per ciascun elemento del testo generato.
            
            ### Livelli di analisi:
            
            - **Token**: Analizza ogni singolo token (unità di testo più piccola riconosciuta dal modello)
            - **Parola**: Analizza la confidenza media per ogni parola
            - **Frase**: Analizza la confidenza media per ogni frase
            
            ### Interpretazione dei colori:
            
            - **Verde**: Alta confidenza (il modello è molto sicuro)
            - **Giallo/Arancione**: Media confidenza
            - **Rosso**: Bassa confidenza (il modello è incerto)
            
            ### Note sull'API:
            
            - Richiede una API key OpenAI valida
            - Utilizza il parametro `logprobs=True` disponibile in alcuni modelli
            """)
    
    return app

# Avvia l'applicazione
if __name__ == "__main__":
    app = create_interface()
    app.launch(share=True)  # Aggiunto 'share=True' per generare un link pubblico
