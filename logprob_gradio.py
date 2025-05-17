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

def split_into_sentences(text: str) -> List[str]:
    """Divide il testo in frasi."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s for s in sentences if s.strip()]

def group_tokens_into_sentences(text: str, tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Raggruppa i token in frasi e calcola la confidenza media per ogni frase."""
    sentences = split_into_sentences(text)
    sentence_data = []
    current_token_index = 0
    
    for sentence in sentences:
        if not sentence.strip():
            continue
        
        sentence_tokens = []
        current_sentence_text = ''
        found_full_sentence = False
        
        while current_token_index < len(tokens) and not found_full_sentence:
            token = tokens[current_token_index]
            current_sentence_text += token['token']
            sentence_tokens.append(token)
            current_token_index += 1
            
            if (sentence in current_sentence_text) or sentence.strip().endswith(current_sentence_text.strip()):
                found_full_sentence = True
        
        if sentence_tokens:
            avg_logprob = sum(token['logprob'] for token in sentence_tokens) / len(sentence_tokens)
            confidence = logprob_to_confidence(avg_logprob)
            
            sentence_data.append({
                'text': sentence,
                'confidence': confidence
            })
    
    return sentence_data

# Funzioni principali
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

def analyze_confidence(api_key: str, model: str, prompt: str) -> Union[List[Dict[str, Any]], Dict[str, str]]:
    """Analizza la confidenza delle frasi generate dal modello."""
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
        
        # Raggruppa i token in frasi
        sentences = group_tokens_into_sentences(text, logprobs)
        
        return sentences
    except Exception as e:
        return {"error": f"Errore: {str(e)}"}

def format_results(result: Union[List[Dict[str, Any]], Dict[str, str]]) -> str:
    """Formatta i risultati dell'analisi in HTML."""
    if isinstance(result, dict) and "error" in result:
        return f"<div style='color: red; padding: 10px; border-left: 4px solid red; background-color: #ffeeee;'><strong>Errore:</strong> {result['error']}</div>"
    
    sentences = result
    html = "<h2>Risultati dell'analisi</h2>"
    
    for i, sentence in enumerate(sentences):
        color = get_confidence_color(sentence['confidence'])
        label = get_confidence_label(sentence['confidence'])
        
        html += f"""
        <div style='margin: 10px 0; padding: 10px; border-radius: 4px; color: {color}; background-color: {color}15; border-left: 4px solid {color};'>
            <strong>Frase {i + 1}:</strong> {sentence['text']}
            <span style='margin-left: 10px; font-size: 0.9em;'>({sentence['confidence']:.2f}% - {label})</span>
        </div>
        """
    
    return html

# Funzioni per Gradio
def run_analysis(api_key: str, model: str, prompt: str) -> str:
    """Funzione principale per l'analisi."""
    result = analyze_confidence(api_key, model, prompt)
    return format_results(result)

# Interfaccia Gradio
def create_interface():
    with gr.Blocks(title="Sentence Confidence Analyzer") as app:
        gr.Markdown("# Sentence Confidence Analyzer")
        
        with gr.Box():
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
        
        with gr.Box():
            gr.Markdown("## Seleziona Modello")
            model_select = gr.Dropdown(
                choices=[
                    "gpt-4o-2024-08-06",
                    "gpt-4o-mini-2024-07-18",
                    "gpt-4-turbo-2024-04-09",
                    "gpt-3.5-turbo-0125"
                ],
                value="gpt-4o-2024-08-06",
                label="Modello"
            )
        
        with gr.Box():
            gr.Markdown("## Prompt")
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Inserisci il tuo prompt",
                value="Descrivi le tecniche di scavo e i principali ritrovamenti archeologici del sito di Göbekli Tepe in Turchia, confrontandoli con quelli di Gunung Padang in Indonesia.",
                lines=5
            )
        
        analyze_btn = gr.Button("Analizza Confidenza")
        results_html = gr.HTML()
        
        # Collegamento per l'analisi con indicatore di caricamento incorporato
        analyze_btn.click(
            run_analysis,
            inputs=[api_key_input, model_select, prompt_input],
            outputs=[results_html],
            show_progress=True
        )
    
    return app

# Avvia l'applicazione
if __name__ == "__main__":
    app = create_interface()
    app.launch()
