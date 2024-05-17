import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Ładowanie modelu i tokenizera T5
@st.cache_resource
def load_model():
    model_name = 't5-large'
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Funkcja do tłumaczenia tekstu
def translate_text(text, source_lang='en', target_lang='de'):
    input_text = f'translate {source_lang} to {target_lang}: {text}'
    st.write(f"Input text for model: {input_text}")  # Debugowanie
    inputs = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
    st.write(f"Tokenized input: {inputs}")  # Debugowanie

    # Generowanie tłumaczenia
    outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    st.write(f"Model outputs: {outputs}")  # Debugowanie

    # Dekodowanie wyniku
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write(f"Translated text: {translated_text}")  # Debugowanie
    return translated_text

# Interfejs użytkownika Streamlit
st.title("Translator from English to German")

english_text = st.text_area("Enter text in English", height=150)
if st.button("Translate"):
    if english_text.strip() == "":
        st.warning("Please enter some text to translate.")
    else:
        with st.spinner("Translating..."):
            try:
                translated_text = translate_text(english_text)
                st.success("Translation completed!")
                st.text_area("Translated text in German", translated_text, height=150)
            except Exception as e:
                st.error(f"Translation failed: {e}")

st.write("This application uses the T5 model to translate text from English to German.")
