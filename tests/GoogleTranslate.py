from googletrans import Translator

async def google_translate(source_sentences, target_language):
    translator = Translator()
    res = []
    try:
        translations = await translator.translate(source_sentences, dest=target_language)
        for translation in translations:
            res.append(translation.text)
        return res
    except Exception as e:
        print(f"Error: {e}")