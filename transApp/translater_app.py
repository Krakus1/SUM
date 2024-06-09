import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer, MarianMTModel, MarianTokenizer

# Globalna zmienna do włączania/wyłączania debugowania
DEBUG = False

# Mapa wyboru modeli
model_map = {
    "Google": "t5-base",
    "Helsinki-NLP": "Helsinki-NLP/opus-mt-en-de"
}

# Funkcja do ładowania modelu i tokenizera
@st.cache_resource
def load_model(model_name):
    if model_name == 't5-base':
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
    elif model_name == 'Helsinki-NLP/opus-mt-en-de':
        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)
    else:
        raise ValueError("Model not supported")
    return model, tokenizer

# Funkcja do tłumaczenia tekstu
def translate_text(text, model, tokenizer, model_name):
    if model_name == 't5-base':
        input_text = f'translate English to German: {text}'
    elif model_name == 'Helsinki-NLP/opus-mt-en-de':
        input_text = text
    else:
        raise ValueError("Model not supported")

    if DEBUG:
        st.write(f"Input text for model: {input_text}")  # Debugowanie

    inputs = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)

    if DEBUG:
        st.write(f"Tokenized input: {inputs}")  # Debugowanie

    # Generowanie tłumaczenia
    outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)

    if DEBUG:
        st.write(f"Model outputs: {outputs}")  # Debugowanie

    # Dekodowanie wyniku
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if DEBUG:
        st.write(f"Translated text: {translated_text}")  # Debugowanie

    return translated_text

# Interfejs użytkownika Streamlit
st.image('https://store-images.s-microsoft.com/image/apps.50108.14056881113182363.a95f37ff-7b9d-43ff-8072-2df8de461120.41ab5e02-0179-47ac-b4fb-6bf5ab010ca2?h=210', caption='Translate be better')
st.title("Translator from English to German")
st.info("Aplikacja służy do przetłumaczenia słów/zwrotów angielskich na jezyk niemiecki z możliwością wyboru modelu :)")

# Opcja wyboru modelu przez użytkownika
model_choice = st.selectbox("Choose a translation model", ["Helsinki-NLP", "Google"])

# Mapowanie wyboru na nazwę modelu
model_name = model_map[model_choice]

# Ładowanie wybranego modelu i tokenizera
model, tokenizer = load_model(model_name)

# Pole do wprowadzania tekstu w języku angielskim
english_text = st.text_area("Enter text in English", height=150)

# Przycisk do tłumaczenia tekstu
if st.button("Translate"):
    if english_text.strip() == "":
        st.warning("Please enter some text to translate.")
    else:
        with st.spinner("Translating..."):
            try:
                translated_text = translate_text(english_text, model, tokenizer, model_name)
                st.success("Translation completed!")
                st.text_area("Translated text in German", translated_text, height=150)
            except Exception as e:
                st.error(f"Translation failed: {e}")

st.write("This app is created by S23942")
