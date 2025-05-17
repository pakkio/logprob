# Sentence Confidence Analyzer

## Overview

Sentence Confidence Analyzer is a web tool that allows you to analyze the confidence level with which an OpenAI language model generates each sentence in a response. This tool provides significant assistance in identifying potential hallucinations (inaccurate information) in AI-generated texts.

![image](https://github.com/user-attachments/assets/e4613415-9f14-4e5c-b66e-88ad8e0994b8)


## Key Features

- **Sentence-level confidence analysis**: Visualizes the average confidence level for each generated sentence
- **Color coding**: Sentences are colored from green (high confidence) to red (low confidence)
- **Multiple model compatibility**: Supports GPT-4o, GPT-4o-mini, GPT-4-turbo, and GPT-3.5-turbo
- **API connection testing**: Verifies the validity of your API key before analysis
- **Intuitive interface**: Easy to use even for non-technical users

## How It Works

The tool uses the "log probabilities" (logprobs) provided by the OpenAI API to calculate how "certain" the model is about the words it generates. Specifically:

1. Sends a request to the OpenAI API with the `logprobs=true` parameter
2. Groups tokens (words or parts of words) into complete sentences
3. Calculates the average confidence for each sentence
4. Displays sentences colored according to their confidence level

## Prerequisites

- A modern web browser
- A valid OpenAI API key
- Internet access

## Installation

No installation required. The tool works directly in your browser:

1. Save the HTML code in a file with `.html` extension
2. Open the file with a web browser

Alternatively, you can host the file on a web server to make it accessible online.

## Usage

1. **Enter your API key**: Input your OpenAI API key in the dedicated field
2. **Verify the connection**: Use the "Test Connection" button to verify the validity of your key
3. **Select a model**: Choose the OpenAI model you want to use (GPT-4o recommended)
4. **Enter your prompt**: Write or paste your prompt text in the text box
5. **Analyze**: Click "Analyze Confidence" to start the process

The results will display each sentence from the response with:
- A color indicating the confidence level
- A numerical confidence percentage
- A qualitative label (Very High, High, Medium, Low, Very Low)

## Interpreting the Results

- **Bright green (>95%)**: The model is extremely confident about this sentence
- **Light green (85-95%)**: High confidence, generally reliable
- **Yellow (70-85%)**: Medium confidence, may contain inaccuracies
- **Orange (50-70%)**: Low confidence, higher risk of inaccuracies
- **Red (<50%)**: Very low confidence, high probability of hallucinations

## Limitations

- Works only with the OpenAI API (does not support other providers like Anthropic Claude or Google Gemini)
- Requires that the model supports the `logprobs` parameter
- Confidence percentages reflect the model's certainty, not necessarily factual accuracy
- Sentence-level analysis is an approximation and may not capture nuances at the individual word level

## Technical Notes

The tool uses pure JavaScript without external dependencies to communicate with the OpenAI API. The main logic:

```javascript
// Convert logprob to confidence percentage
function logprobToConfidence(logprob) {
    return Math.exp(logprob) * 100;
}

// Group tokens into sentences
function groupTokensIntoSentences(text, tokens) {
    // [grouping code]
}
```

## Usage Examples

- **Verification of complex historical facts**: Identify which parts of a response about historical events might be less reliable
- **Analysis of scientific documents**: Determine the confidence level on specific scientific statements
- **Model comparison**: Compare the "certainty" level of different models on the same question
- **AI education**: Visually demonstrate how language models evaluate their own confidence

## Authors

[Your name]

## License

This tool is available under the MIT license.
