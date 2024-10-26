import random

def augment_text(text):
    words = text.split()
    augmented_words = []
    
    for word in words:
        if random.random() < 0.1:  # 10% chance to add a synonym
            augmented_words.append(get_synonym(word))
        else:
            augmented_words.append(word)
    
    return ' '.join(augmented_words)

def get_synonym(word):
    # This is a placeholder function. In a real-world scenario,
    # you would use a dictionary or an API to get actual synonyms.
    synonyms = {
        'happy': ['joyful', 'content', 'pleased'],
        'sad': ['unhappy', 'melancholy', 'gloomy'],
        'big': ['large', 'huge', 'enormous'],
        'small': ['tiny', 'little', 'miniature'],
    }
    return random.choice(synonyms.get(word, [word]))
