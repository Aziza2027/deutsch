from deep_translator import GoogleTranslator

def trnaslate_text(text_list):
    # Use any translator you like, in this example GoogleTranslator
    text_list_tanslated = []
    for text in text_list:
        translated = GoogleTranslator(source='de', target='en').translate(text)
        text_list_tanslated.append(translated)
    # translated = GoogleTranslator(source='de', target='en').translate(text)  # output -> Weiter so, du bist großartig
    return text_list_tanslated